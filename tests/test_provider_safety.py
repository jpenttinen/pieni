"""Offline regressions for provider completion and continuation safety."""

import copy
import importlib.util
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import pieni
from tests import test_pieni as fixtures


class CompletionStatusTests(fixtures.TempWorkspaceCase):
    def test_buffered_chat_rejects_unsuccessful_finish_reasons(self):
        for reason in ("length", "content_filter", "error"):
            for kind in ("custom", "chat", "openrouter"):
                with self.subTest(reason=reason, kind=kind):
                    body = fixtures.chat_reply("partial answer")
                    body.choices[0].finish_reason = reason
                    client = (fixtures.FakeSendClient(body) if kind == "openrouter"
                              else fixtures.FakeChatClient(body))
                    provider = pieni.Provider("test", "m", kind, client, streaming=False)
                    with self.assertRaisesRegex(pieni.ProviderError, reason):
                        provider.complete([])

    def test_buffered_responses_rejects_incomplete_failed_and_error_replies(self):
        for status, error, expected in (
                ("incomplete", None, "max_output_tokens"),
                ("failed", {"message": "upstream failed"}, "upstream failed"),
                ("completed", {"message": "upstream error"}, "upstream error")):
            with self.subTest(status=status, error=error):
                body = fixtures.responses_reply("partial answer")
                body.status = status
                body.error = error
                body.incomplete_details = {"reason": "max_output_tokens"}
                client = fixtures.FakeResponsesClient(body)
                with self.assertRaisesRegex(pieni.ProviderError, expected):
                    pieni.call_responses(client, "m", [], None, streaming=False)

    def test_unsuccessful_buffered_reply_executes_no_tools_and_saves_no_assistant(self):
        for kind in ("custom", "chat", "openrouter", "responses"):
            with self.subTest(kind=kind):
                call = ("call_1", "write", '{"path":"new.txt","content":"partial"}')
                if kind == "responses":
                    body = fixtures.responses_reply(calls=[call])
                    body.status = "incomplete"
                    body.incomplete_details = {"reason": "max_output_tokens"}
                    client = fixtures.FakeResponsesClient(body)
                else:
                    body = fixtures.chat_reply(calls=[call])
                    body.choices[0].finish_reason = "length"
                    client = (fixtures.FakeSendClient(body) if kind == "openrouter"
                              else fixtures.FakeChatClient(body))
                agent, _, store, output = self.make_agent([])
                agent.provider = pieni.Provider("test", "m", kind, client, streaming=False)
                with mock.patch.object(agent.tools, "run") as run:
                    self.assertFalse(agent.run_task("write a file"))
                run.assert_not_called()
                self.assertFalse((self.workspace / "new.txt").exists())
                self.assertEqual(agent.messages, [{"role": "user", "content": "write a file"}])
                _, saved = store.resume("scripted", "scripted-model", str(self.workspace))
                self.assertEqual(saved, agent.messages)
                self.assertTrue(any(line.startswith("error:") for line in output))


def native_output():
    return [
        {"type": "reasoning", "id": "rs_1", "summary": [],
         "encrypted_content": "opaque reasoning"},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "phase": "commentary", "content": [
             {"type": "output_text", "text": "I will read the file.", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read",
         "arguments": '{"path":"a.txt"}', "status": "completed"},
    ]


def native_reply(items, text):
    return SimpleNamespace(output=items, output_text=text, status="completed",
                           usage={"input_tokens": 10, "output_tokens": 4})


class ResponsesReplayTests(fixtures.TempWorkspaceCase):
    def test_native_sequence_survives_tool_continuation_and_database_resume(self):
        self.write(self.workspace / "a.txt", "file body")
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                expected = native_output()
                bodies = [native_reply(copy.deepcopy(expected), "I will read the file."),
                          native_reply([{"type": "message", "id": "msg_2", "role": "assistant",
                                         "status": "completed", "phase": "final_answer", "content": [
                                             {"type": "output_text", "text": "done", "annotations": []}]}],
                                       "done"),
                          native_reply([], "resumed")]
                requests = []

                def create(**kwargs):
                    requests.append(kwargs)
                    body = bodies.pop(0)
                    return fixtures.fake_responses_stream(body) if kwargs["stream"] else body

                client = SimpleNamespace(responses=SimpleNamespace(create=create))
                agent, _, store, _ = self.make_agent([])
                agent.provider = pieni.Provider("openai", "m", "responses", client,
                                                streaming=streaming)
                self.assertTrue(agent.run_task("read a.txt"))
                continuation = requests[1]["input"]
                self.assertEqual(continuation[1:4], expected)
                self.assertEqual(continuation[4], {"type": "function_call_output", "call_id": "call_1",
                                                  "output": "a.txt (lines 1-1 of 1)\nfile body"})
                # The canonical text/call remains useful locally but is never sent twice.
                self.assertEqual(len(continuation), 5)
                _, saved = store.resume("scripted", "scripted-model", str(self.workspace))
                self.assertEqual(saved[1]["responses_output"], expected)
                self.assertEqual(saved[1]["tool_calls"][0]["id"], "call_1")
                agent.messages = saved
                self.assertTrue(agent.run_task("continue"))
                self.assertEqual(requests[2]["input"][1:4], expected)
                self.assertEqual(sum(item.get("type") == "function_call_output"
                                     for item in requests[2]["input"]), 1)

    def test_resumed_native_output_omits_null_sdk_defaults_without_changing_history(self):
        expected = native_output()
        legacy = copy.deepcopy(expected)
        legacy[0].update(status=None, content=None)
        legacy[1]["phase"] = None
        expected[1].pop("phase")
        agent, _, store, _ = self.make_agent([])
        message = {"role": "assistant", "content": "I will read the file.",
                   "responses_output": legacy}
        agent.remember(message)
        _, saved = store.resume("scripted", "scripted-model", str(self.workspace))
        client = fixtures.FakeResponsesClient(native_reply([], "done"))
        pieni.call_responses(client, "m", saved, None, streaming=False)
        self.assertEqual(client.requests[0]["input"], expected)
        self.assertEqual(saved[0]["responses_output"], legacy)
        self.assertEqual(message["responses_output"], legacy)

    def test_serializes_sdk_models_with_all_json_fields(self):
        expected = native_output()
        items = [SimpleNamespace(model_dump=mock.Mock(return_value=copy.deepcopy(item)))
                 for item in expected]
        # The adapter reads attributes as well as using the model's native dump.
        for item, fields in zip(items, expected):
            item.__dict__.update(fields)
        body = native_reply(items, "I will read the file.")
        reply = pieni.call_responses(fixtures.FakeResponsesClient(body), "m", [], None,
                                     streaming=False)
        for item in items:
            item.model_dump.assert_called_once_with(mode="json", by_alias=True)
        message = pieni.assistant_message(reply)
        self.assertEqual(json.loads(json.dumps(message))["responses_output"], expected)
        self.assertEqual(pieni.to_responses_input([message]), expected)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "the openai SDK is not installed")
    def test_actual_sdk_output_models_are_json_serializable_and_replayed(self):
        from openai.types.responses import (
            ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem,
        )
        fields = native_output()
        items = [ResponseReasoningItem(**fields[0]), ResponseOutputMessage(**fields[1]),
                 ResponseFunctionToolCall(**fields[2])]
        expected = [{key: value for key, value in item.model_dump(mode="json", by_alias=True).items()
                     if value is not None} for item in items]
        reply = pieni.call_responses(fixtures.FakeResponsesClient(
            native_reply(items, "I will read the file.")), "m", [], None, streaming=False)
        message = json.loads(json.dumps(pieni.assistant_message(reply)))
        self.assertEqual(pieni.to_responses_input([message]), expected)

    def test_old_session_conversion_remains_compatible(self):
        old = {"role": "assistant", "content": "old reply", "tool_calls": [
            {"id": "call_1", "name": "read", "arguments": '{"path":"a.txt"}'}]}
        self.assertEqual(pieni.to_responses_input([old]), [
            {"role": "assistant", "content": "old reply"},
            {"type": "function_call", "call_id": "call_1", "name": "read",
             "arguments": '{"path":"a.txt"}'},
        ])

    def test_compaction_omits_native_replay_but_retains_visible_text_and_tools(self):
        message = {"role": "assistant", "content": "I will read the file.",
                   "responses_output": native_output(), "tool_calls": [
                       {"id": "call_1", "name": "read", "arguments": '{"path":"a.txt"}'}]}
        history = json.loads(pieni.compaction_history([message]))
        self.assertNotIn("responses_output", history[0])
        self.assertNotIn("opaque reasoning", json.dumps(history))
        self.assertEqual(history[0]["content"], message["content"])
        self.assertEqual(history[0]["tool_calls"][0]["arguments"], {"path": "a.txt"})


if __name__ == "__main__":
    unittest.main()
