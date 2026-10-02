"""Optional live smoke tests: real provider/network, with any provider charges.

This file is skipped unless you opt in explicitly, for example:

    PIENI_LIVE_PROVIDER=deepseek PIENI_LIVE_MODEL=deepseek-chat \
    DEEPSEEK_API_KEY=... python3 -m unittest tests.test_live

The default suite is the offline `python3 -m unittest discover`.
"""

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path

import pieni

PROVIDER = os.environ.get("PIENI_LIVE_PROVIDER", "")
MODEL = os.environ.get("PIENI_LIVE_MODEL", "")


@unittest.skipUnless(PROVIDER and MODEL,
                     "set PIENI_LIVE_PROVIDER and PIENI_LIVE_MODEL to run live tests")
class LiveSmokeTest(unittest.TestCase):
    """Tool execution and resume with the provider and model you selected."""

    def run_task(self, directory, prompt, streaming=True):
        previous = os.getcwd()
        os.chdir(directory)
        stdout = StringIO()
        try:
            with redirect_stdout(stdout):
                code = pieni.main([PROVIDER, "-m", MODEL, "-r", prompt,
                                   "--permissions", "auto",
                                   "--streaming" if streaming else "--no-streaming"])
        finally:
            os.chdir(previous)
        text = stdout.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("Tokens:", text)
        self.assertRegex(text, r"\d+\.\d seconds")
        return text

    def test_headless_task_round_trip(self):
        prompt = "Run the bash command `echo pieni-live-ok` and answer with its output only."
        for streaming in (True, False):
            with self.subTest(streaming=streaming), tempfile.TemporaryDirectory() as directory:
                text = self.run_task(directory, prompt, streaming)
                self.assertRegex(text, r"bash\(.*\) -> ok,")
                self.assertIn("pieni-live-ok", text)
                self.assertTrue((Path(directory) / ".pieni" / "pieni.db").is_file())

    def test_file_tools_and_session_resume(self):
        prompt = (
            'Use write to create greeting.txt containing exactly "Hei, 世界!\\n". '
            'Then use read to inspect it, and edit to replace "Hei" with "Moi". '
            'Finally use bash to run "cat greeting.txt". Use all four named tools. '
            'Reply with the final file contents.'
        )
        with tempfile.TemporaryDirectory() as directory:
            text = self.run_task(directory, prompt)
            path = Path(directory) / "greeting.txt"
            self.assertEqual(path.read_text(encoding="utf-8"), "Moi, 世界!\n", text)
            for tool in ("write", "read", "edit", "bash"):
                self.assertRegex(text, rf"{tool}\(.*\) -> ok,")
            database = Path(directory) / ".pieni" / "pieni.db"
            with sqlite3.connect(database) as connection:
                before = connection.execute("SELECT COUNT(*) FROM messages WHERE role='tool'").fetchone()[0]
            resumed = self.run_task(directory, "Without tools, name the file we just edited and its final contents.")
            self.assertIn("resumed a session", resumed)
            self.assertIn("greeting.txt", resumed)
            self.assertIn("Moi, 世界!", resumed)
            self.assertEqual(path.read_text(encoding="utf-8"), "Moi, 世界!\n")
            with sqlite3.connect(database) as connection:
                after = connection.execute("SELECT COUNT(*) FROM messages WHERE role='tool'").fetchone()[0]
            self.assertEqual(before, after, resumed)  # historical calls were not executed again

    def test_compaction_preserves_constraints_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with closing(pieni.build_provider(PROVIDER, MODEL)) as provider:
                with closing(pieni.Store(workspace / pieni.DB_PATH)) as store:
                    session = store.start_session(PROVIDER, MODEL, directory)
                    agent = pieni.Agent(provider, store, session, pieni.SYSTEM_PROMPT, [], workspace,
                                        pieni.Permissions("auto", workspace))
                    for message in (
                            {"role": "user", "content": "Fix parser.py. Preserve CRLF and do not change the public API."},
                            {"role": "assistant", "content": "Running tests.", "tool_calls": [
                                {"id": "t1", "name": "bash", "arguments": '{"command":"python3 -m unittest"}'}]},
                            pieni.tool_message("t1", "bash", "exit code 1\n" + "." * 40000 +
                                               "\nFAIL: test_crlf\nExpected CRLF, got LF.\nFAILED (failures=1)"),
                            {"role": "assistant", "content": "No files changed. test_crlf still fails. Next inspect parser.py."}):
                        agent.remember(message)
                    with redirect_stdout(StringIO()):
                        self.assertTrue(agent.compact())
                    self.assertEqual(len(agent.messages), 1)
                    summary = agent.messages[0]["content"]
                    for fact in ("parser.py", "CRLF", "API", "test_crlf"):
                        self.assertIn(fact, summary)
                    active = list(agent.messages)
            with closing(pieni.Store(workspace / pieni.DB_PATH)) as reopened:
                self.assertEqual(reopened.resume(PROVIDER, MODEL, directory)[1], active)
                self.assertEqual(reopened.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 5)


if __name__ == "__main__":
    unittest.main()
