"""Offline reasoning controls, provider continuation, and direct shell commands."""

import json
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import pieni
from tests import test_pieni as fixtures
from tests.test_streaming import chunk


class ReasoningTests(fixtures.TempWorkspaceCase):
    def sdk_agent(self, kind, bodies, streaming=False, reasoning="high"):
        agent, _, store, output = self.make_agent([])
        client_type = {"responses": fixtures.FakeResponsesClient,
                       "openrouter": fixtures.FakeSendClient}.get(kind, fixtures.FakeChatClient)
        client = client_type(None)
        send = (client.responses.create if kind == "responses" else
                client.chat.send if kind == "openrouter" else client.chat.completions.create)
        pending = iter(bodies)

        def scripted(**kwargs):
            client.reply = next(pending)
            return send(**kwargs)

        if kind == "responses":
            client.responses.create = scripted
        elif kind == "openrouter":
            client.chat.send = scripted
        else:
            client.chat.completions.create = scripted
        agent.provider = pieni.Provider(kind, "m", kind, client, streaming=streaming,
                                        reasoning=reasoning)
        return agent, client, store, output

    def test_cli_values_and_invalid_inputs(self):
        self.assertIsNone(pieni.parse_args([]).reasoning)
        for effort in ("default", "none", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.assertEqual(pieni.parse_args(["--reasoning", effort]).reasoning, effort)
        for args in (["--reasoning"], ["--reasoning", ""], ["--reasoning", "huge"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                pieni.parse_args(args)

    def test_ini_precedence_and_default_reset(self):
        load = lambda **kwargs: pieni.load_config(cwd=self.workspace, home=self.home, **kwargs)
        self.assertEqual(load()["reasoning"], "default")
        self.write(self.home / ".pieni/pieni.ini", "[pieni]\nreasoning = low\n")
        self.write(self.workspace / "pieni.ini", "[pieni]\nmodel = m\n")
        self.assertEqual(load()["reasoning"], "low")
        self.write(self.workspace / "pieni.ini", "[pieni]\nreasoning = high\n")
        self.assertEqual(load()["reasoning"], "high")
        self.assertEqual(load(cli_reasoning="none")["reasoning"], "none")
        self.assertEqual(load(cli_reasoning="default")["reasoning"], "default")
        self.write(self.workspace / "pieni.ini", "[pieni]\nreasoning = default\n")
        self.assertEqual(load()["reasoning"], "default")

    def test_invalid_ini_and_programmatic_values_report_origin(self):
        for value in ("huge", "", "高"):
            path = self.write(self.workspace / "pieni.ini", f"[pieni]\nreasoning = {value}\n")
            with self.assertRaises(pieni.ConfigError) as caught:
                pieni.load_config(cwd=self.workspace, home=self.home)
            self.assertIn(str(path), str(caught.exception))
            self.assertIn("reasoning", str(caught.exception))
        with mock.patch("pieni.load_sdk") as sdk, self.assertRaises(pieni.ConfigError):
            pieni.build_provider("openai", "m", reasoning="huge")
        sdk.assert_not_called()

    def test_cli_passes_effort_in_every_interface(self):
        self.in_directory(self.workspace)
        path = self.write(self.workspace / "pieni.ini",
                          "[pieni]\nprovider = openai\nmodel = m\nreasoning = low\n")
        for interface in (["--streaming"], ["--run", "hello"], ["--prompt", "hello"]):
            for flags, expected in (([], "low"), (["--reasoning", "high"], "high"),
                                    (["--reasoning", "default"], "default")):
                with self.subTest(interface=interface, flags=flags):
                    provider = SimpleNamespace(
                        name="openai", model="m", streaming=False,
                        complete=mock.Mock(return_value=fixtures.reply_from(text="done")),
                        close=mock.Mock())
                    with mock.patch("pieni.config_paths", return_value=[path]):
                        with mock.patch("pieni.build_provider", return_value=provider) as build:
                            with mock.patch("builtins.input", return_value="/quit"), redirect_stdout(StringIO()):
                                self.assertEqual(pieni.main(interface + flags), 0)
                    build.assert_called_once_with("openai", "m", streaming=True, reasoning=expected)

    def test_effort_request_fields_for_every_provider_and_streaming_mode(self):
        for kind in ("responses", "openrouter", "chat", "custom"):
            for streaming in (False, True):
                for effort in ("default", "none", "high"):
                    with self.subTest(kind=kind, streaming=streaming, effort=effort):
                        body = (fixtures.responses_reply("done") if kind == "responses"
                                else fixtures.chat_reply("done"))
                        agent, client, _, _ = self.sdk_agent(kind, [body], streaming, effort)
                        self.assertTrue(agent.run_task("hello"))
                        request = client.requests[0]
                        self.assertEqual(request["stream"], streaming)
                        if effort == "default":
                            for key in ("reasoning", "reasoning_effort", "extra_body"):
                                self.assertNotIn(key, request)
                        elif kind in ("responses", "openrouter"):
                            self.assertEqual(request["reasoning"], {"effort": effort})
                        elif kind == "chat":
                            state = "disabled" if effort == "none" else "enabled"
                            self.assertEqual(request["extra_body"], {"thinking": {"type": state}})
                            if effort == "none":
                                self.assertNotIn("reasoning_effort", request)
                            else:
                                self.assertEqual(request["reasoning_effort"], effort)
                        else:
                            self.assertEqual(request["reasoning_effort"], effort)

    def test_runtime_controls_next_task_and_compaction_without_writing_config(self):
        path = self.write(self.workspace / "pieni.ini", "[pieni]\nreasoning = low\n")
        agent, client, _, output = self.sdk_agent(
            "responses", [fixtures.responses_reply("done"), fixtures.responses_reply("summary"),
                          fixtures.responses_reply("again")], reasoning="low")
        agent.handle_command("/reasoning")
        self.assertEqual(output[-1], "reasoning: low")
        agent.handle_command("/reasoning high")
        self.assertEqual(client.requests, [])
        self.assertTrue(agent.run_task("hello"))
        self.assertEqual(client.requests[-1]["reasoning"], {"effort": "high"})
        agent.handle_command("/reasoning max")
        self.assertTrue(agent.compact())
        self.assertEqual(client.requests[-1]["reasoning"], {"effort": "max"})
        self.assertNotIn("tools", client.requests[-1])
        agent.handle_command("/reasoning default")
        self.assertTrue(agent.run_task("hello again"))
        self.assertNotIn("reasoning", client.requests[-1])
        self.assertEqual(path.read_text(), "[pieni]\nreasoning = low\n")
        for command in ("/reasoning huge", "/reasoning high extra"):
            agent.handle_command(command)
            self.assertEqual(agent.provider.reasoning, "default")

    def test_provider_rejection_is_reported_without_retry(self):
        agent, client, _, output = self.sdk_agent("custom", [RuntimeError("unsupported effort")])
        self.assertFalse(agent.run_task("hello"))
        self.assertEqual(len(client.requests), 1)
        self.assertIn("unsupported effort", "\n".join(output))

    def test_reasoning_survives_tool_calls_and_resume_but_not_compaction(self):
        secret = "完整 reasoning\n" * 200
        details = [{"type": "reasoning.text", "text": secret, "signature": "signed",
                    "format": "anthropic-claude-v1", "index": 0}]
        for kind in ("chat", "openrouter"):
            for streaming in (False, True):
                with self.subTest(kind=kind, streaming=streaming):
                    bodies = [fixtures.chat_reply(calls=[("call_1", "bash", '{"command":"true"}')]),
                              fixtures.chat_reply("done"), fixtures.chat_reply("resumed"),
                              fixtures.chat_reply("summary")]
                    for body in bodies[:2]:
                        if kind == "chat":
                            body.choices[0].message.reasoning_content = secret
                        else:
                            body.choices[0].message.reasoning = secret
                            body.choices[0].message.reasoning_details = details
                    agent, client, store, _ = self.sdk_agent(kind, bodies, streaming)
                    self.assertTrue(agent.run_task("hello"))
                    key = "reasoning_content" if kind == "chat" else "reasoning"
                    self.assertEqual(client.requests[1]["messages"][2][key], secret)
                    _, saved = store.resume("scripted", "scripted-model", str(self.workspace))
                    self.assertEqual(saved[-1][key], secret)
                    if kind == "openrouter":
                        self.assertEqual(saved[-1]["reasoning_details"], details)
                    agent.messages = saved
                    self.assertTrue(agent.run_task("continue"))
                    self.assertEqual(client.requests[2]["messages"][2][key], secret)
                    self.assertEqual(sum(m["role"] == "tool" for m in agent.messages), 1)
                    self.assertTrue(agent.compact())
                    self.assertNotIn(secret, json.dumps(client.requests[-1]))
                    self.assertNotIn(key, pieni.compaction_history(saved))

    def test_openrouter_stream_preserves_order_and_complete_fields(self):
        blocks = [{"type": "reasoning.text", "text": "part ", "index": 0, "signature": None},
                  {"type": "reasoning.text", "text": "two", "index": 0, "signature": "sig"},
                  {"type": "reasoning.encrypted", "data": "opaque", "index": 1}]
        events = [chunk({"reasoning": "part ", "reasoning_details": blocks[:1]}),
                  chunk({"reasoning": "two", "reasoning_details": blocks[1:], "content": "done"}),
                  chunk(finish="stop")]
        stream = fixtures.FakeStream(events)
        client = SimpleNamespace(chat=SimpleNamespace(send=mock.Mock(return_value=stream)))
        reply = pieni.call_openrouter(client, "m", [], None)
        saved = pieni.assistant_message(reply)
        self.assertEqual(saved["reasoning"], "part two")
        self.assertEqual(saved["reasoning_details"], blocks)
        self.assertTrue(stream.closed)

    def test_incomplete_reasoning_stream_saves_no_assistant_or_tool_call(self):
        agent, _, store, _ = self.make_agent([])
        stream = fixtures.FakeStream([chunk({"reasoning_content": "partial", "content": "partial"})])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=mock.Mock(return_value=stream))))
        agent.provider = pieni.Provider("deepseek", "m", "chat", client)
        self.assertFalse(agent.run_task("hello"))
        self.assertEqual(agent.messages, [{"role": "user", "content": "hello"}])
        _, saved = store.resume("scripted", "scripted-model", str(self.workspace))
        self.assertEqual(saved, agent.messages)
        self.assertTrue(stream.closed)


class ShellShortcutTests(fixtures.TempWorkspaceCase):
    def test_output_timing_and_no_conversation_or_model_use(self):
        agent, provider, store, output = self.make_agent([])
        with mock.patch("pieni.time.monotonic", side_effect=[2.0, 2.012]):
            result = agent.run_shell("printf '世界\\n'")
        self.assertTrue(result.ok)
        self.assertEqual(output, ["世界", "!printf '世界\\n' -> ok, 12 ms"])
        self.assertEqual(agent.messages, [])
        self.assertEqual(provider.requests, [])
        self.assertIsNone(store.last_message_id(agent.session_id))

    def test_working_directory_and_separate_shells(self):
        agent, _, _, output = self.make_agent([])
        agent.run_shell("cd /tmp")
        agent.run_shell("pwd")
        self.assertEqual(output[-2], str(self.workspace))

    def test_mv_and_cat_with_spaces_and_unicode(self):
        self.write(self.workspace / "old.text", "世界\n")
        agent, _, _, output = self.make_agent([])
        agent.run_shell("mv old.text 'new name.txt'")
        agent.run_shell("cat 'new name.txt'")
        self.assertFalse((self.workspace / "old.text").exists())
        self.assertEqual(output[-2], "世界")

    def test_stderr_exit_code_empty_output_and_unknown_command(self):
        agent, _, _, output = self.make_agent([])
        agent.run_shell("printf boom >&2; exit 3")
        self.assertIn("boom", output[-2])
        self.assertRegex(output[-1], r"error \(exit code 3\), \d+ ms$")
        agent.run_shell("true")
        self.assertEqual(output[-2], "(no output)")
        agent.run_shell("pieni_command_that_does_not_exist")
        self.assertIn("exit code 127", output[-1])

    def test_guard_denial_approval_and_yolo(self):
        for mode, approved in (("auto", False), ("auto", True), ("yolo", False)):
            with self.subTest(mode=mode, approved=approved):
                approve = mock.Mock(return_value=approved)
                agent, _, _, output = self.make_agent([], mode=mode, approve=approve)
                with mock.patch("pieni.subprocess.run", return_value=SimpleNamespace(
                        stdout="", stderr="", returncode=0)) as run:
                    agent.run_shell("rm -rf build")
                self.assertEqual(run.called, approved or mode == "yolo")
                if mode == "yolo":
                    approve.assert_not_called()
                else:
                    approve.assert_called_once()
                self.assertIn("ok" if run.called else "denied", output[-1])

    def test_empty_commands_do_not_execute(self):
        agent, _, _, output = self.make_agent([])
        with mock.patch.object(agent.tools, "run") as run:
            for command in ("", "   "):
                agent.run_shell(command)
        run.assert_not_called()
        self.assertEqual(output, ["usage: !COMMAND", "usage: !COMMAND"])

    def test_timeout_oserror_and_interrupt_return_to_prompt(self):
        for error, expected in ((subprocess.TimeoutExpired("sleep", 60), "timeout after 60s"),
                                (OSError("cannot spawn"), "OSError"),
                                (KeyboardInterrupt(), "interrupted")):
            with self.subTest(error=error):
                agent, provider, store, output = self.make_agent([])
                with mock.patch("pieni.subprocess.run", side_effect=error) as run:
                    with mock.patch("builtins.input", side_effect=["!sleep 999", "/quit"]):
                        with redirect_stdout(StringIO()):
                            self.assertEqual(pieni.run_interactive(agent), 0)
                run.assert_called_once()
                self.assertIn(expected, "\n".join(output))
                self.assertRegex(output[-2], r"error \(.+\), \d+ ms$")
                self.assertEqual(output[-1], "bye")
                self.assertEqual(provider.requests, [])
                self.assertIsNone(store.last_message_id(agent.session_id))

    def test_output_limit_and_invalid_encoding(self):
        agent, _, _, output = self.make_agent([])
        agent.run_shell("head -c 70000 /dev/zero | tr '\\0' x")
        self.assertIn("truncated", output[-2])
        agent.run_shell("printf '\\377hello'")
        self.assertIn("hello", output[-2])
        self.assertIn("\ufffd", output[-2])

    def test_interactive_routes_only_prefixed_commands_and_continues_tasks(self):
        agent, provider, _, output = self.make_agent([fixtures.reply_from(text="answer")])
        with mock.patch("builtins.input", side_effect=["!printf shortcut", "cat /tmp/example.c", "/quit"]):
            with redirect_stdout(StringIO()):
                self.assertEqual(pieni.run_interactive(agent), 0)
        self.assertIn("shortcut", output)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(agent.messages[0], {"role": "user", "content": "cat /tmp/example.c"})
        self.assertNotIn("shortcut", json.dumps(agent.messages))
