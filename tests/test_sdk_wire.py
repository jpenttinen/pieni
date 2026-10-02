"""Wire-format tests against the real provider SDKs, using a loopback server.

These tests make no external requests and spend no money: a local HTTP server on
127.0.0.1 captures what each SDK actually sends, and replies with a canned body
that the SDK then parses. That is how the request and response shapes in
pieni.py are checked against the installed `openai` and `openrouter` packages.

Run them with the project virtualenv on the path (the default offline suite,
`tests.test_pieni`, does not need the SDKs):

    .venv/bin/python -m unittest tests.test_sdk_wire

They skip when a needed SDK is not importable.
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pieni

HAVE_OPENAI = importlib.util.find_spec("openai") is not None
HAVE_OPENROUTER = importlib.util.find_spec("openrouter") is not None


def responses_body(text="", calls=()):
    """A minimal Responses API body; extra fields keep real clients happy."""
    output = [{"type": "function_call", "id": "fc_1", "call_id": call_id, "name": name,
               "arguments": arguments, "status": "completed"}
              for call_id, name, arguments in calls]
    if text:
        output.append({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "test-model",
        "status": "completed", "output": output, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [], "error": None, "incomplete_details": None,
        "instructions": None, "metadata": {}, "temperature": 1, "top_p": 1,
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    }


def chat_body(text="", calls=(), system_fingerprint="fp_test"):
    """A minimal Chat Completions body; the openrouter model requires the fingerprint."""
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [
            {"id": call_id, "type": "function", "index": index,
             "function": {"name": name, "arguments": arguments}}
            for index, (call_id, name, arguments) in enumerate(calls)
        ]
    return {
        "id": "chatcmpl_1", "object": "chat.completion", "created": 0, "model": "test-model",
        "system_fingerprint": system_fingerprint,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
    }


def stream_payload(body):
    """Encode canned replies as real SSE, including fragmented tool arguments."""
    events = []
    if body.get("object") == "response":
        for item in body["output"]:
            for part in item.get("content", []):
                text = part.get("text", "")
                for fragment in (text[:2], text[2:]):
                    events.append({"type": "response.output_text.delta", "delta": fragment,
                                   "item_id": item["id"], "output_index": 0,
                                   "content_index": 0, "sequence_number": len(events)})
        events.append({"type": "response.completed", "response": body,
                       "sequence_number": len(events)})
    else:
        base = {key: body[key] for key in ("id", "created", "model", "system_fingerprint")}
        base["object"] = "chat.completion.chunk"
        message = body["choices"][0]["message"]
        delta = {"role": "assistant", "content": message.get("content")}
        for key in pieni.REASONING_FIELDS:
            if key in message:
                delta[key] = message[key]
        fragments = []
        for index, call in enumerate(message.get("tool_calls", [])):
            arguments = call["function"]["arguments"]
            fragments.append({"index": index, "id": call["id"], "type": "function",
                              "function": {"name": call["function"]["name"],
                                           "arguments": arguments[:2]}})
        if fragments:
            delta["tool_calls"] = fragments
        events.append(dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]))
        if fragments:
            tail = [{"index": index, "function": {"arguments": call["function"]["arguments"][2:]}}
                    for index, call in enumerate(message["tool_calls"])]
            events.append(dict(base, choices=[{"index": 0, "delta": {"tool_calls": tail},
                                               "finish_reason": None}]))
        events.append(dict(base, choices=[{"index": 0, "delta": {},
                                           "finish_reason": "tool_calls" if fragments else "stop"}]))
        events.append(dict(base, choices=[], usage=body["usage"]))
    data = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    return (data + "data: [DONE]\n\n").encode("utf-8")


class Recorder:
    """Captured requests and the canned replies to send back."""

    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.requests = []

    def next_body(self):
        return self.bodies.pop(0) if self.bodies else {"error": "no canned body left"}


class LoopbackServer:
    """A local HTTP server that records POST bodies from the SDKs."""

    def __init__(self, recorder):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 (required name)
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length).decode("utf-8")
                outer.recorder.requests.append(
                    {"path": self.path, "json": json.loads(raw or "{}")})
                body = outer.recorder.next_body()
                streaming = outer.recorder.requests[-1]["json"].get("stream", False)
                payload = stream_payload(body) if streaming else json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep the test output quiet
                pass

        self.recorder = recorder
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        return False


@unittest.skipUnless(HAVE_OPENAI, "the openai SDK is not installed")
class OpenAITests(unittest.TestCase):
    """The Responses path, through the real openai SDK."""

    MESSAGES = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "name": "read", "arguments": '{"path": "a.txt"}'}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "read", "content": "file body"},
    ]

    def test_request_shape_and_response_parsing(self):
        recorder = Recorder([
            responses_body(calls=[("call_2", "bash", '{"command": "ls"}')]),
            responses_body(text="all done"),
        ])
        with LoopbackServer(recorder) as server:
            # The real SDK reads OPENAI_API_KEY and OPENAI_BASE_URL from the process
            # environment, so point it at the loopback server.
            environment = {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": server.url}
            with mock.patch.dict(os.environ, environment):
                provider = pieni.build_provider("openai", "test-model", environment)
                try:
                    first = provider.complete(self.MESSAGES)
                    second = provider.complete(self.MESSAGES + [
                        pieni.assistant_message(first),
                        pieni.tool_message("call_2", "bash", "exit code 0"),
                    ])
                finally:
                    provider.close()

        first_request = recorder.requests[0]["json"]
        self.assertEqual(recorder.requests[0]["path"], "/v1/responses")
        self.assertTrue(first_request["stream"])
        self.assertEqual(first_request["model"], "test-model")
        self.assertEqual(first_request["instructions"], "sys")
        self.assertNotIn("system", [item.get("role") for item in first_request["input"]])
        self.assertEqual(first_request["input"][0], {"role": "user", "content": "hi"})
        self.assertEqual(first_request["input"][1],
                         {"type": "function_call", "call_id": "call_1", "name": "read",
                          "arguments": '{"path": "a.txt"}'})
        self.assertEqual(first_request["input"][2],
                         {"type": "function_call_output", "call_id": "call_1",
                          "output": "file body"})
        tool = first_request["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "read")
        self.assertIn("parameters", tool)
        self.assertNotIn("function", tool)

        self.assertEqual(first.tool_calls[0].name, "bash")
        self.assertEqual(first.tool_calls[0].arguments, '{"command": "ls"}')
        self.assertFalse(first.estimated)
        self.assertEqual((first.input_tokens, first.output_tokens), (11, 7))
        self.assertEqual(second.text, "all done")
        self.assertIn({"type": "function_call_output", "call_id": "call_2", "output": "exit code 0"},
                      recorder.requests[1]["json"]["input"])

    def test_nonstreaming_response_with_real_sdk(self):
        from openai import OpenAI
        recorder = Recorder([responses_body(text="buffered")])
        with LoopbackServer(recorder) as server:
            with OpenAI(api_key="test-key", base_url=server.url) as client:
                reply = pieni.call_responses(client, "m", self.MESSAGES, None, streaming=False)
        self.assertEqual(reply.text, "buffered")
        self.assertFalse(recorder.requests[0]["json"]["stream"])


    def test_live_tool_schema_is_accepted_by_the_sdk_types(self):
        """The tool schema pieni builds must match the SDK's expected fields."""
        from openai.types.responses.function_tool_param import FunctionToolParam
        from openai.types.shared_params.function_definition import FunctionDefinition

        spec = pieni.responses_tools()[0]
        self.assertEqual(set(FunctionToolParam.__annotations__) & {"type", "name", "parameters"},
                         {"type", "name", "parameters"})
        self.assertEqual(set(spec) & set(FunctionToolParam.__annotations__), set(spec) - {"strict"})
        chat_spec = pieni.chat_tools()[0]["function"]
        self.assertEqual(set(chat_spec) <= set(FunctionDefinition.__annotations__), True)


@unittest.skipUnless(HAVE_OPENAI, "the openai SDK is not installed")
class DeepSeekTests(unittest.TestCase):
    """The Chat Completions path, through the real openai SDK."""

    def test_request_shape_and_tool_result_round_trip(self):
        recorder = Recorder([
            chat_body(calls=[("call_1", "write", '{"path": "a", "content": "b"}')]),
            chat_body(text="written"),
        ])
        with LoopbackServer(recorder) as server:
            with mock.patch.object(pieni, "DEEPSEEK_BASE_URL", server.url):
                provider = pieni.build_provider("deepseek", "deepseek-chat",
                                                {"DEEPSEEK_API_KEY": "test-key"})
            try:
                first = provider.complete([{"role": "system", "content": "sys"},
                                           {"role": "user", "content": "hi"}])
                second = provider.complete([
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    pieni.assistant_message(first),
                    pieni.tool_message("call_1", "write", "created a"),
                ])
            finally:
                provider.close()

        request = recorder.requests[0]["json"]
        self.assertTrue(request["stream"])
        self.assertEqual(request["stream_options"], {"include_usage": True})
        self.assertEqual(request["model"], "deepseek-chat")
        self.assertEqual(request["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(request["tools"][0]["type"], "function")
        self.assertEqual(request["tools"][0]["function"]["name"], "read")
        self.assertIn("parameters", request["tools"][0]["function"])
        self.assertEqual(first.tool_calls[0].id, "call_1")
        self.assertEqual((first.input_tokens, first.output_tokens), (13, 5))

        follow_up = recorder.requests[1]["json"]["messages"]
        self.assertEqual(follow_up[2]["tool_calls"][0],
                         {"id": "call_1", "type": "function",
                          "function": {"name": "write", "arguments": '{"path": "a", "content": "b"}'}})
        self.assertEqual(follow_up[3]["role"], "tool")
        self.assertEqual(follow_up[3]["tool_call_id"], "call_1")
        self.assertEqual(follow_up[3]["content"], "created a")
        self.assertEqual(second.text, "written")


@unittest.skipUnless(HAVE_OPENROUTER, "the openrouter SDK is not installed")
class OpenRouterTests(unittest.TestCase):
    """The OpenRouter path, through the real openrouter SDK."""

    def test_request_shape_through_chat_send(self):
        recorder = Recorder([chat_body(text="openrouter answer")])
        with LoopbackServer(recorder) as server:
            from openrouter import OpenRouter
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                with OpenRouter(api_key="test-key", server_url=server.url) as client:
                    reply = pieni.call_openrouter(
                        client, "vendor/model",
                        [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "hi"}],
                        pieni.chat_tools())

        request = recorder.requests[0]["json"]
        self.assertTrue(request["stream"])
        self.assertEqual(request["model"], "vendor/model")
        self.assertEqual(request["messages"][0], {"role": "system", "content": "sys"})
        tool = request["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "read")
        self.assertIn("parameters", tool["function"])
        self.assertEqual(reply.text, "openrouter answer")
        self.assertFalse(reply.tool_calls)

    def test_openrouter_client_is_closed_by_pieni(self):
        recorder = Recorder([chat_body(text="ok")])
        with LoopbackServer(recorder) as server:
            from openrouter import OpenRouter
            original = pieni.load_sdk
            created = []

            class TrackingOpenRouter:
                """A real OpenRouter client plus a record of context-manager use."""

                def __init__(self, **kwargs):
                    self.inner = OpenRouter(server_url=server.url, **kwargs)
                    self.entered = False
                    self.exited = False
                    created.append(self)

                def __enter__(self):
                    self.entered = True
                    return self.inner.__enter__()

                def __exit__(self, *exc_info):
                    self.exited = True
                    return self.inner.__exit__(*exc_info)

            def loader(module_name, attribute_name):
                if module_name == "openrouter":
                    return TrackingOpenRouter
                return original(module_name, attribute_name)

            with mock.patch.object(pieni, "load_sdk", loader):
                provider = pieni.build_provider("openrouter", "vendor/model",
                                                {"OPENROUTER_API_KEY": "test-key"})
                client = created[-1]
                self.assertTrue(client.entered)
                self.assertFalse(client.exited)
                self.assertEqual(provider.complete([{"role": "user", "content": "hi"}]).text, "ok")
                provider.close()
                self.assertTrue(client.exited)  # the SDK context manager really exited
                provider.close()  # and closing twice stays harmless
        self.assertTrue(created[-1].exited)


class ReasoningWireTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_OPENAI, "the openai SDK is not installed")
    def test_effort_fields_through_responses_custom_and_deepseek_sdks(self):
        from openai import OpenAI

        for kind in ("responses", "custom", "chat"):
            body = responses_body(text="done") if kind == "responses" else chat_body(text="done")
            recorder = Recorder([body] * 6)
            with LoopbackServer(recorder) as server:
                with OpenAI(api_key="test", base_url=server.url) as client:
                    for streaming in (False, True):
                        for effort in ("default", "none", "high"):
                            with self.subTest(kind=kind, streaming=streaming, effort=effort):
                                provider = pieni.Provider(kind, "m", kind, client,
                                                          streaming=streaming, reasoning=effort)
                                self.assertEqual(provider.complete([
                                    {"role": "user", "content": "hello"}]).text, "done")
                                request = recorder.requests[-1]["json"]
                                if effort == "default":
                                    for key in ("reasoning", "reasoning_effort", "thinking"):
                                        self.assertNotIn(key, request)
                                elif kind == "responses":
                                    self.assertEqual(request["reasoning"], {"effort": effort})
                                elif kind == "custom":
                                    self.assertEqual(request["reasoning_effort"], effort)
                                else:
                                    self.assertEqual(request["thinking"], {
                                        "type": "disabled" if effort == "none" else "enabled"})
                                    if effort == "none":
                                        self.assertNotIn("reasoning_effort", request)
                                    else:
                                        self.assertEqual(request["reasoning_effort"], effort)

    @unittest.skipUnless(HAVE_OPENAI, "the openai SDK is not installed")
    def test_deepseek_full_reasoning_round_trip_with_real_sdk(self):
        from openai import OpenAI

        secret = "完整 reasoning\n" * 200
        for streaming in (False, True):
            first_body = chat_body(calls=[("call_1", "bash", '{"command":"true"}')])
            first_body["choices"][0]["message"]["reasoning_content"] = secret
            recorder = Recorder([first_body, chat_body(text="done")])
            with LoopbackServer(recorder) as server:
                with OpenAI(api_key="test", base_url=server.url) as client:
                    provider = pieni.Provider("deepseek", "m", "chat", client,
                                              streaming=streaming, reasoning="high")
                    messages = [{"role": "user", "content": "hello"}]
                    first = provider.complete(messages)
                    provider.complete(messages + [pieni.assistant_message(first),
                                      pieni.tool_message("call_1", "bash", "ok")])
            self.assertEqual(recorder.requests[1]["json"]["messages"][1]["reasoning_content"], secret)

    @unittest.skipUnless(HAVE_OPENROUTER, "the openrouter SDK is not installed")
    def test_openrouter_effort_and_reasoning_blocks_round_trip_with_real_sdk(self):
        from openrouter import OpenRouter

        details = [{"type": "reasoning.text", "text": "thinking", "signature": "sig",
                    "index": 0, "format": "anthropic-claude-v1"},
                   {"type": "reasoning.encrypted", "data": "opaque", "index": 1,
                    "format": "anthropic-claude-v1"}]
        for streaming in (False, True):
            first_body = chat_body(calls=[("call_1", "bash", '{"command":"true"}')])
            first_body["choices"][0]["message"].update(reasoning="thinking", reasoning_details=details)
            recorder = Recorder([first_body, chat_body(text="done"), chat_body(text="default")])
            with LoopbackServer(recorder) as server:
                with OpenRouter(api_key="test", server_url=server.url) as client:
                    provider = pieni.Provider("openrouter", "m", "openrouter", client,
                                              streaming=streaming, reasoning="high")
                    messages = [{"role": "user", "content": "hello"}]
                    first = provider.complete(messages)
                    provider.reasoning = "none"
                    provider.complete(messages + [pieni.assistant_message(first),
                                      pieni.tool_message("call_1", "bash", "ok")])
                    provider.reasoning = "default"
                    provider.complete(messages)
            self.assertEqual(recorder.requests[0]["json"]["reasoning"], {"effort": "high"})
            self.assertEqual(recorder.requests[1]["json"]["reasoning"], {"effort": "none"})
            followup = recorder.requests[1]["json"]["messages"][1]
            self.assertEqual(followup["reasoning"], "thinking")
            self.assertEqual(followup["reasoning_details"], details)
            self.assertNotIn("reasoning", recorder.requests[2]["json"])


@unittest.skipUnless(HAVE_OPENAI, "the openai SDK is not installed")
class EndToEndTests(unittest.TestCase):
    """The whole program — loop, tools, database, display — with the real SDK."""

    def test_headless_task_with_the_real_sdk(self):
        recorder = Recorder([
            responses_body(calls=[("call_1", "bash",
                                   json.dumps({"command": "echo real-sdk-ok"}))]),
            responses_body(text="the command printed real-sdk-ok"),
        ])
        with LoopbackServer(recorder) as server:
            with tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(pieni, "config_paths", return_value=[]):
                    environment = {"OPENAI_API_KEY": "test-key",
                                   "OPENAI_BASE_URL": server.url}
                    with mock.patch.dict(os.environ, environment):
                        stdout = io.StringIO()
                        previous = os.getcwd()
                        try:
                            os.chdir(directory)
                            with redirect_stdout(stdout):
                                code = pieni.main(["openai", "-m", "test-model",
                                                   "-r", "print the marker", "--reasoning", "high"])
                        finally:
                            os.chdir(previous)
                database = os.path.join(directory, ".pieni", "pieni.db")
                self.assertTrue(os.path.isfile(database))

        text = stdout.getvalue()
        self.assertEqual(code, 0, text)
        for request in recorder.requests:
            self.assertEqual(request["json"]["reasoning"], {"effort": "high"})
        # The bash tool really ran, and its result went back to the model.
        self.assertRegex(text, r'bash\(command="echo real-sdk-ok"\) -> ok, \d+ ms')
        self.assertIn("the command printed real-sdk-ok", text)
        self.assertRegex(text, r"Tokens: 36 \| Context: ~[\d,]+ / 1,000,000 \(\d+\.\d%\) "
                               r"\| \d+\.\d seconds")
        second_input = recorder.requests[1]["json"]["input"]
        outputs = [item for item in second_input if item.get("type") == "function_call_output"]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["call_id"], "call_1")
        self.assertIn("real-sdk-ok", outputs[0]["output"])


if __name__ == "__main__":
    unittest.main()
