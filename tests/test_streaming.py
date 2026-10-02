"""Offline streaming configuration, assembly, display, and failure-path tests."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import pieni
from tests import test_pieni as fixtures


def chunk(delta=None, finish=None, usage=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
            "usage": usage}


class StreamingConfigTests(fixtures.TempWorkspaceCase):
    def test_boolean_ini_values(self):
        for value, expected in (("true", True), ("YES", True), ("on", True), ("1", True),
                                ("false", False), ("NO", False), ("off", False), ("0", False)):
            with self.subTest(value=value):
                self.write(self.workspace / "pieni.ini", f"[pieni]\nstreaming = {value}\n")
                settings = pieni.load_config(cwd=self.workspace, home=self.home)
                self.assertIs(settings["streaming"], expected)

    def test_user_local_cli_precedence_including_false(self):
        self.write(self.home / ".pieni" / "pieni.ini", "[pieni]\nstreaming = false\n")
        self.assertFalse(pieni.load_config(cwd=self.workspace, home=self.home)["streaming"])
        self.write(self.workspace / "pieni.ini", "[pieni]\nstreaming = true\n")
        self.assertTrue(pieni.load_config(cwd=self.workspace, home=self.home)["streaming"])
        for value in (False, True):
            settings = pieni.load_config(cwd=self.workspace, home=self.home, cli_streaming=value)
            self.assertIs(settings["streaming"], value)

    def test_invalid_streaming_value_identifies_file(self):
        for value in ("sometimes", "", "启用"):
            with self.subTest(value=value):
                path = self.write(self.workspace / "pieni.ini", f"[pieni]\nstreaming = {value}\n")
                with self.assertRaises(pieni.ConfigError) as caught:
                    pieni.load_config(cwd=self.workspace, home=self.home)
                self.assertIn(str(path), str(caught.exception))
                self.assertIn("streaming", str(caught.exception))

    def test_cli_flags_have_unset_default_and_are_mutually_exclusive(self):
        self.assertIsNone(pieni.parse_args([]).streaming)
        self.assertTrue(pieni.parse_args(["--streaming"]).streaming)
        self.assertFalse(pieni.parse_args(["--no-streaming"]).streaming)
        with redirect_stdout(StringIO()), mock.patch("sys.stderr", new=StringIO()):
            with self.assertRaises(SystemExit):
                pieni.parse_args(["--streaming", "--no-streaming"])

    def test_cli_passes_resolved_streaming_setting_to_provider(self):
        self.in_directory(self.workspace)
        self.write(self.workspace / "pieni.ini",
                   "[pieni]\nprovider = openai\nmodel = m\nstreaming = false\n")
        for flags, expected in (([], False), (["--streaming"], True), (["--no-streaming"], False)):
            with self.subTest(flags=flags):
                provider = fixtures.ScriptedProvider([fixtures.reply_from(text="done")])
                with mock.patch("pieni.config_paths", return_value=[self.workspace / "pieni.ini"]):
                    with mock.patch("pieni.build_provider", return_value=provider) as build:
                        with redirect_stdout(StringIO()):
                            self.assertEqual(pieni.main(["-r", "hi"] + flags), 0)
                build.assert_called_once_with("openai", "m", streaming=expected, reasoning="default")

    def test_all_provider_paths_default_to_streaming_and_allow_disable(self):
        sdks = {"openai": SimpleNamespace(OpenAI=fixtures.FakeOpenAI),
                "openrouter": SimpleNamespace(OpenRouter=fixtures.FakeOpenRouter)}
        environment = {"OPENAI_API_KEY": "test", "DEEPSEEK_API_KEY": "test",
                       "OPENROUTER_API_KEY": "test"}
        with mock.patch.dict("sys.modules", sdks):
            for name in ("openai", "deepseek", "openrouter", "http://localhost:30000"):
                with self.subTest(provider=name):
                    provider = pieni.build_provider(name, "any-model", environment)
                    self.assertTrue(provider.streaming)
                    provider.close()
                    provider = pieni.build_provider(name, "any-model", environment, streaming=False)
                    self.assertFalse(provider.streaming)
                    provider.close()


class StreamingAssemblyTests(unittest.TestCase):
    def test_chat_fragments_multiple_tools_unicode_reasoning_and_final_usage(self):
        events = [
            chunk({"content": "你好", "reasoning_content": "think ", "tool_calls": [
                {"index": 1, "id": "call_b", "function": {"name": "write", "arguments": '{"path":'}},
                {"index": 0, "id": "call_a", "function": {"name": "read", "arguments": '{"path":'}}]}),
            chunk({"content": " مرحبا", "reasoning_content": "first", "tool_calls": [
                {"index": 0, "function": {"arguments": '"文件.txt"}'}},
                {"index": 1, "function": {"arguments": '"b", "content":"مرحبا"}'}}]}),
            chunk(finish="tool_calls"),
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 9}},
        ]
        for kind in ("chat", "custom", "openrouter"):
            with self.subTest(kind=kind):
                stream = fixtures.FakeStream(events)
                send = mock.Mock(return_value=stream)
                client = SimpleNamespace(chat=SimpleNamespace(
                    send=send, completions=SimpleNamespace(create=send)))
                provider = pieni.Provider(kind, "m", kind, client)
                output = []
                reply = provider.complete([{"role": "user", "content": "hi"}], on_text=output.append)
                self.assertTrue(send.call_args.kwargs["stream"])
                self.assertEqual(output, ["你好", " مرحبا"])
                self.assertEqual(reply.text, "你好 مرحبا")
                self.assertEqual(reply.thinking, "think first")
                self.assertEqual([call.id for call in reply.tool_calls], ["call_a", "call_b"])
                self.assertEqual(json.loads(reply.tool_calls[0].arguments), {"path": "文件.txt"})
                self.assertEqual(json.loads(reply.tool_calls[1].arguments)["content"], "مرحبا")
                self.assertEqual((reply.input_tokens, reply.output_tokens), (12, 9))
                self.assertFalse(reply.estimated)
                self.assertTrue(stream.closed)

    def test_responses_final_object_preserves_tools_reasoning_usage(self):
        body = fixtures.responses_reply("你好", [("call_1", "read", '{"path":"a"}')])
        body.output.append({"type": "reasoning", "summary": [{"text": "consider files"}]})
        stream = fixtures.FakeStream([
            {"type": "response.output_item.added", "item": {"type": "function_call"}},
            {"type": "response.function_call_arguments.delta", "delta": '{"path":'},
            {"type": "response.output_text.delta", "delta": "你"},
            {"type": "response.output_text.delta", "delta": "好"},
            {"type": "response.completed", "response": body},
        ])
        client = SimpleNamespace(responses=SimpleNamespace(create=mock.Mock(return_value=stream)))
        output = []
        reply = pieni.call_responses(client, "m", [], None, on_text=output.append)
        self.assertEqual(output, ["你", "好"])
        self.assertEqual(reply.text, "你好")
        self.assertEqual(reply.tool_calls[0].arguments, '{"path":"a"}')
        self.assertEqual(reply.thinking, "consider files")
        self.assertEqual((reply.input_tokens, reply.output_tokens), (50, 10))
        self.assertTrue(stream.closed)

    def test_missing_usage_is_estimated(self):
        for call, client in ((pieni.call_chat_completions,
                              fixtures.FakeChatClient(fixtures.chat_reply("hi", usage=False))),
                             (pieni.call_openrouter,
                              fixtures.FakeSendClient(fixtures.chat_reply("hi", usage=False))),
                             (pieni.call_responses,
                              fixtures.FakeResponsesClient(fixtures.responses_reply("hi", usage=False)))):
            reply = call(client, "m", [{"role": "user", "content": "hello"}], None)
            self.assertTrue(reply.estimated)
            self.assertGreater(reply.input_tokens, 0)
            self.assertGreater(reply.output_tokens, 0)

    def test_nonstreaming_uses_buffered_reply_and_never_calls_callback(self):
        for call, client in ((pieni.call_chat_completions, fixtures.FakeChatClient(fixtures.chat_reply("hi"))),
                             (pieni.call_openrouter, fixtures.FakeSendClient(fixtures.chat_reply("hi"))),
                             (pieni.call_responses, fixtures.FakeResponsesClient(fixtures.responses_reply("hi")))):
            callback = mock.Mock()
            reply = call(client, "m", [], None, streaming=False, on_text=callback)
            self.assertEqual(reply.text, "hi")
            self.assertFalse(client.requests[0]["stream"])
            self.assertNotIn("stream_options", client.requests[0])
            callback.assert_not_called()

    def test_chat_truncation_error_invalid_tool_index_and_empty_stream(self):
        cases = [[], [chunk({"content": "partial"})], [chunk(finish="length")],
                 [chunk(finish="content_filter")], [{"error": {"message": "upstream failed"}}],
                 [chunk({"tool_calls": [{"function": {"name": "read"}}]})],
                 [chunk({"tool_calls": [{"index": 0, "function": {"name": "read"}}]}),
                  chunk(finish="tool_calls")]]
        for events in cases:
            with self.subTest(events=events):
                stream = fixtures.FakeStream(events)
                with self.assertRaises(pieni.ProviderError):
                    pieni.collect_chat_stream(stream)
                self.assertTrue(stream.closed)

    def test_responses_failed_incomplete_error_or_missing_completion(self):
        for events in ([], [{"type": "response.output_text.delta", "delta": "partial"}],
                       [{"type": "response.failed"}], [{"type": "response.incomplete"}],
                       [{"type": "error", "message": "bad request"}]):
            stream = fixtures.FakeStream(events)
            with self.assertRaises(pieni.ProviderError):
                pieni.collect_responses_stream(stream)
            self.assertTrue(stream.closed)

    def test_stream_closes_on_network_failure_or_interrupt(self):
        for collector in (pieni.collect_chat_stream, pieni.collect_responses_stream):
            for error in (RuntimeError("network reset"), KeyboardInterrupt()):
                stream = fixtures.FakeStream([error])
                with self.assertRaises(type(error)):
                    collector(stream)
                self.assertTrue(stream.closed)


class StreamingLoopTests(fixtures.TempWorkspaceCase):
    def make_streaming_agent(self, events):
        agent, _, store, output = self.make_agent([])
        stream = fixtures.FakeStream(events)
        send = mock.Mock(return_value=stream)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)))
        agent.provider = pieni.Provider("deepseek", "m", "chat", client)
        raw = []
        agent.stream_out = raw.append
        return agent, store, output, raw, stream, send

    def test_streamed_text_displayed_once_and_deepseek_reasoning_saved_for_replay(self):
        agent, store, output, raw, stream, _ = self.make_streaming_agent([
            chunk({"content": "你", "reasoning_content": "inspect files"}),
            chunk({"content": "好 مرحبا"}), chunk(finish="stop"),
        ])
        self.assertTrue(agent.run_task("hello"))
        self.assertEqual(raw, ["你", "好 مرحبا", "\n"])
        self.assertNotIn("你好 مرحبا", output)
        self.assertIn("thinking: inspect files", output)
        self.assertEqual(agent.messages[-1]["content"], "你好 مرحبا")
        _, saved = store.resume(fixtures.ScriptedProvider.name, fixtures.ScriptedProvider.model,
                                str(self.workspace))
        self.assertEqual(saved[-1]["reasoning_content"], "inspect files")
        self.assertNotIn("thinking:", json.dumps(saved))
        self.assertTrue(stream.closed)
        self.assertIn("Tokens: ~", output[-1])

    def test_partial_tool_call_never_executed_or_saved_on_failure(self):
        for failure in (RuntimeError("network reset"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                agent, _, output, raw, stream, send = self.make_streaming_agent([
                    chunk({"content": "working", "tool_calls": [
                        {"index": 0, "id": "call_1", "function": {
                            "name": "write", "arguments": '{"path":"new.txt","content":"x"}'}}]}),
                    failure,
                ])
                with mock.patch.object(agent.tools, "run") as run:
                    self.assertFalse(agent.run_task("go"))
                run.assert_not_called()
                self.assertEqual(len(agent.messages), 1)  # user only, no partial assistant
                self.assertEqual(raw, ["working", "\n"])
                self.assertTrue(stream.closed)
                self.assertEqual(send.call_count, 1)  # no fallback or silent retry
                self.assertIn("Tokens: ~", output[-1])
                self.assertNotIn("Tokens: ~0 |", output[-1])
                self.assertTrue(any("error:" in line or "interrupted" in line for line in output))

    def test_complete_tool_call_runs_only_after_stream_closes(self):
        agent, _, output, raw, stream, send = self.make_streaming_agent([
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {
                "name": "write", "arguments": '{"path":"new.txt",'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"content":"你好"}'}}]}),
            chunk(finish="tool_calls"),
        ])
        final = fixtures.fake_chat_stream(fixtures.chat_reply("done"))
        send.side_effect = [stream, final]
        original = agent.tools.run

        def run(name, arguments):
            self.assertTrue(stream.closed)
            return original(name, arguments)

        with mock.patch.object(agent.tools, "run", side_effect=run):
            self.assertTrue(agent.run_task("create file"))
        self.assertEqual((self.workspace / "new.txt").read_text(encoding="utf-8"), "你好")
        self.assertEqual(raw, ["done", "\n"])
        self.assertTrue(any(" -> ok," in line for line in output))
        continuation = send.call_args_list[1].kwargs["messages"]
        self.assertEqual(continuation[-1]["tool_call_id"], "call_1")

    def test_streaming_compaction_failure_keeps_context(self):
        agent, _, output, raw, stream, send = self.make_streaming_agent([
            chunk({"content": "partial summary"}), RuntimeError("network reset"),
        ])
        agent.remember({"role": "user", "content": "important context"})
        original = list(agent.messages)
        self.assertFalse(agent.compact())
        self.assertEqual(agent.messages, original)
        self.assertEqual(raw, ["partial summary", "\n"])
        self.assertIn("context is unchanged", output[-1])
        self.assertTrue(stream.closed)
        self.assertNotIn("tools", send.call_args.kwargs)

    def test_streaming_compaction_saves_completed_summary(self):
        agent, _, output, raw, stream, send = self.make_streaming_agent([
            chunk({"content": "working summary"}), chunk(finish="stop"),
        ])
        agent.remember({"role": "user", "content": "older context"})
        self.assertTrue(agent.compact())
        self.assertEqual(raw, ["working summary", "\n"])
        self.assertEqual(len(agent.messages), 1)
        self.assertIn("working summary", agent.messages[0]["content"])
        self.assertTrue(stream.closed)
        self.assertTrue(send.call_args.kwargs["stream"])
        self.assertNotIn("tools", send.call_args.kwargs)



if __name__ == "__main__":
    unittest.main()
