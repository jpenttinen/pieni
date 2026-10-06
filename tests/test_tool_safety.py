"""Offline regression tests for invalid tool text and process cleanup."""

import io
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pieni


class ToolSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.permissions = pieni.Permissions("auto", self.workspace, tempdir=self.root / "tmp")
        self.toolbox = pieni.Toolbox(self.workspace, self.permissions)

    def test_invalid_utf8_write_preserves_existing_file_and_mode(self):
        path = self.workspace / "important.txt"
        path.write_bytes(b"valuable existing content\n")
        path.chmod(0o640)
        outcome = self.toolbox.run("write", {"path": path.name, "content": "\ud800"})
        self.assertFalse(outcome.ok)
        self.assertIn("valid UTF-8", outcome.output)
        self.assertEqual(path.read_bytes(), b"valuable existing content\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_invalid_utf8_edit_preserves_existing_file(self):
        path = self.workspace / "important.txt"
        path.write_bytes(b"valuable existing content\r\n")
        outcome = self.toolbox.run("edit", {"path": path.name,
                                           "old_text": "existing", "new_text": "\udfff"})
        self.assertFalse(outcome.ok)
        self.assertEqual(path.read_bytes(), b"valuable existing content\r\n")

    def test_invalid_utf8_new_file_does_not_create_parent_directories(self):
        outcome = self.toolbox.run("write", {"path": "new/file.txt", "content": "\ud800"})
        self.assertFalse(outcome.ok)
        self.assertFalse((self.workspace / "new").exists())

    def test_nul_path_and_command_return_tool_errors(self):
        for name, arguments in (
            ("read", {"path": "invalid\0name"}),
            ("write", {"path": "invalid\0name", "content": "value"}),
            ("edit", {"path": "invalid\0name", "old_text": "old", "new_text": "new"}),
            ("bash", {"command": "echo invalid\0command"}),
        ):
            with self.subTest(tool=name):
                outcome = self.toolbox.run(name, arguments)
                self.assertFalse(outcome.ok)
                self.assertIn("NUL", outcome.output)

    def test_unicode_and_nul_file_content_remain_supported(self):
        path = self.workspace / "notes.txt"
        content = "moi 世界\0\n"
        outcome = self.toolbox.run("write", {"path": path.name, "content": content})
        self.assertTrue(outcome.ok, outcome.output)
        self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_rejected_surrogate_is_printable_and_keeps_agent_context_complete(self):
        class Provider:
            streaming = False

            def __init__(self):
                self.replies = iter([
                    pieni.Reply("", [pieni.ToolCall("invalid-write", "write",
                        '{"path":"important.txt","content":"\\ud800"}')], 0, 0, True),
                    pieni.Reply("The invalid write was rejected.", [], 0, 0, True),
                ])

            def complete(self, messages, **options):
                return next(self.replies)

        path = self.workspace / "important.txt"
        path.write_text("valuable content", encoding="utf-8")
        store = pieni.Store(self.root / "pieni.db")
        self.addCleanup(store.close)
        session = store.start_session("fake", "fake", str(self.workspace))
        output = io.BytesIO()
        stdout = io.TextIOWrapper(output, encoding="utf-8", write_through=True)
        try:
            with redirect_stdout(stdout):
                agent = pieni.Agent(Provider(), store, session, "test", [], self.workspace,
                                    self.permissions)
                self.assertTrue(agent.run_task("exercise invalid tool text"))
            rendered = output.getvalue().decode("utf-8")
        finally:
            stdout.detach()
        self.assertIn("\\ud800", rendered)
        self.assertIn("valid UTF-8", rendered)
        self.assertEqual(path.read_text(encoding="utf-8"), "valuable content")
        messages = store.resume("fake", "fake", str(self.workspace))[1]
        self.assertEqual([message["role"] for message in messages],
                         ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[2]["tool_call_id"], "invalid-write")

    def child_command(self, ready, marker, redirected=False):
        script = (
            "import time; from pathlib import Path; "
            f"Path({str(ready)!r}).touch(); time.sleep(2); "
            f"Path({str(marker)!r}).write_text('unexpected late write')"
        )
        command = shlex.quote(sys.executable) + " -c " + shlex.quote(script)
        if redirected:
            command += " >/dev/null 2>&1"
        return command + "; :"  # keep a shell waiting for its child

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_timeout_stops_child_before_it_can_write_later(self):
        for redirected in (False, True):
            with self.subTest(redirected=redirected):
                ready = self.workspace / f"ready-{redirected}"
                marker = self.workspace / f"late-{redirected}"
                outcome = self.toolbox.run("bash", {
                    "command": self.child_command(ready, marker, redirected), "timeout": 1,
                })
                self.assertFalse(outcome.ok)
                self.assertIn("timeout", outcome.detail)
                self.assertTrue(ready.exists(), "child must have started before the timeout")
                time.sleep(1.3)  # beyond the child's scheduled write
                self.assertFalse(marker.exists(), "timed-out child continued running")

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups")
    def test_interruption_stops_and_reaps_shell_and_child(self):
        ready = self.workspace / "ready-interrupt"
        marker = self.workspace / "late-interrupt"
        processes = []

        def interrupt_after_child_starts(process, **options):
            processes.append(process)
            deadline = time.monotonic() + 2
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "child did not start")
            raise KeyboardInterrupt

        with mock.patch("pieni.subprocess.Popen.communicate", new=interrupt_after_child_starts):
            with self.assertRaises(KeyboardInterrupt):
                self.toolbox.run("bash", {"command": self.child_command(ready, marker)})
        self.assertIsNotNone(processes[0].returncode, "shell was not reaped")
        time.sleep(2.2)
        self.assertFalse(marker.exists(), "interrupted child continued running")

    def test_non_posix_timeout_kills_and_reaps_immediate_process(self):
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.communicate.side_effect = subprocess.TimeoutExpired("command", 1)
        with mock.patch("pieni.os.name", "nt"), mock.patch("pieni.os.killpg", create=True) as killpg:
            with mock.patch("pieni.subprocess.Popen", return_value=process) as spawn:
                outcome = self.toolbox.bash("command", timeout=1)
        self.assertFalse(outcome.ok)
        self.assertIn("timeout", outcome.detail)
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        killpg.assert_not_called()
        self.assertFalse(spawn.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
