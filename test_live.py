"""Optional live smoke test: real provider, real network, real charges.

This file is skipped unless you opt in explicitly, for example:

    PIENI_LIVE_PROVIDER=deepseek PIENI_LIVE_MODEL=deepseek-chat \
    DEEPSEEK_API_KEY=... python3 -m unittest test_live

The default suite is the offline `python3 -m unittest test_pieni`.
"""

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pieni

PROVIDER = os.environ.get("PIENI_LIVE_PROVIDER", "")
MODEL = os.environ.get("PIENI_LIVE_MODEL", "")


@unittest.skipUnless(PROVIDER and MODEL,
                     "set PIENI_LIVE_PROVIDER and PIENI_LIVE_MODEL to run live tests")
class LiveSmokeTest(unittest.TestCase):
    """One real headless task with the provider and model you selected."""

    def test_headless_task_round_trip(self):
        prompt = "Run the bash command `echo pieni-live-ok` and answer with its output only."
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            stdout = StringIO()
            try:
                with redirect_stdout(stdout):
                    code = pieni.main([PROVIDER, "-m", MODEL, "-r", prompt])
            finally:
                os.chdir(previous)
            text = stdout.getvalue()
            self.assertEqual(code, 0, text)
            self.assertIn("Tokens:", text)
            self.assertRegex(text, r"\d+\.\d seconds")
            self.assertTrue((Path(directory) / ".pieni" / "pieni.db").is_file())


if __name__ == "__main__":
    unittest.main()
