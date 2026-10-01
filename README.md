# pieni

**v0.1**

A tiny AI coding agent written in Python: one file of about 1,150 lines
(`pieni.py`) plus a small Bash launcher (`pieni`). It is meant for learning how
agents work — read it, run it, fork it, change it.

"pieni" is Finnish and means "small".

## Status

This is the first working release. Headless mode has run successfully against all
four provider paths: OpenAI, DeepSeek, OpenRouter, and a local OpenAI-compatible
server. Interactive mode, approval prompts, and manual compaction are implemented
and covered by the offline tests, but have had less real-terminal use, so treat
them as the least tested part of v0.1.

Pieni supports these providers (set the API key as an environment variable):

- OpenAI (`OPENAI_API_KEY`) — Responses API
- DeepSeek (`DEEPSEEK_API_KEY`) — Chat Completions
- OpenRouter (`OPENROUTER_API_KEY`) — openrouter SDK
- Any custom base URL (`OPENAI_API_KEY` when the server needs one), including
  local servers such as llama.cpp / llama-server, ollama, LM Studio, vLLM —
  Chat Completions

![pieni AI agent](pieni3.jpg "pieni AI agent")

## Install

Create a virtual environment and install the two direct dependencies
(`openai` and `openrouter`; everything else is the standard library):

```console
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip` instead of `.venv/bin/pip`. The `./pieni`
launcher uses `.venv/bin/python` automatically when that file exists, so you do
not have to activate the environment:

```console
./pieni openai -m "gpt-5"
```

## Configuration

Pieni reads INI settings with `configparser`, in this order:

1. `~/.pieni/pieni.ini` — your settings.
2. `pieni.ini` in the directory where you launch Pieni — overrides your settings.
3. Command-line arguments — override both files.

Example:

```ini
[pieni]
provider = deepseek
model = deepseek-chat
permissions = auto
```

Missing files are fine. Malformed INI files, unknown settings, or invalid values
are reported with a clear error. API keys belong in environment variables, never
in these files.

## CLI usage

```console
./pieni                                       # provider and model from pieni.ini
./pieni openai -m "gpt-5"                     # interactive session
./pieni deepseek -m "deepseek-chat"
./pieni openrouter -m "vendor/model"
./pieni http://localhost:30000 -m "qwen3-30b" # local server
./pieni openai -m "gpt-5" --permissions yolo
```

## Headless mode

```console
./pieni openai -m "gpt-5" -r "Explain this repository in five bullets."
```

`-r/--run` runs one task, prints the answer and the task summary, and exits. The
exit code is nonzero when the task could not finish, and actions that would need
approval are denied instead of waiting for input.

Headless mode is the path verified in v0.1 across all providers:

```console
./pieni openai -m "gpt-5" -r "Summarize README.md."
./pieni deepseek -m "deepseek-chat" -r "Summarize README.md."
./pieni openrouter -m "vendor/model" -r "Summarize README.md."
./pieni http://localhost:30000 -m "qwen3-30b" -r "Summarize README.md."
```

Model names are examples only; use whatever your provider or local server offers.

## Commands

```console
/compact        replace older context with a model-written summary
/compact all    clear the context and start a fresh session
/permissions    show permissions; /permissions auto|yolo changes them
/help           concise help
/quit, /exit    leave
```

Ctrl+C interrupts the current task and returns to the prompt; Ctrl+C or Ctrl+D at
an idle prompt exits. Completed actions are not rolled back.

## Tools

The model gets four tools, and Pieni validates their arguments, bounds their
output, and reports failures back to the model:

- `read` — read a UTF-8 file, optionally a line range.
- `write` — create or replace a UTF-8 file.
- `edit` — replace one exact text match; fails when the text is missing or
  ambiguous.
- `bash` — run a shell command in the workspace with a timeout.

## Output

After every tool call, Pieni prints a status line with the outcome and the elapsed
time in milliseconds:

```console
read(path="pieni.py") -> ok, 12 ms
bash(command="python3 -m unittest test_pieni") -> error: exit code 1, 340 ms
```

After every completed, failed, or interrupted task, it prints token usage, context
usage against an assumed 1,000,000-token window, and elapsed time:

```console
Tokens: ~2,400 | Context: ~12,000 / 1,000,000 (1.2%) | 3.1 seconds
```

Token counts come from the provider when it reports usage and are otherwise a
marked text estimate. The 1,000,000-token context window is a display assumption,
not a claim about the selected model.

## Permissions and safety

- `auto` (default) runs shell commands unless the destructive-command guard (DCG)
  matches an obviously destructive pattern (`rm -rf`, `mkfs`, `dd of=…`,
  `shutdown`, `git push --force`, `DROP TABLE`, …). Flagged commands need your
  approval.
- File tools work inside the launch directory and the system temp directory;
  paths outside them need approval, with `..` and symlinks resolved first.
- `yolo` skips approval and guard checks, but not argument validation or timeouts.
- The guard is best-effort pattern matching. It can miss destructive commands and
  flag harmless ones, and it is **not** a sandbox: `bash` runs with your user's
  filesystem and network access.

## Saved data

Conversations — prompts, replies, tool calls, results, and the active context —
are stored in SQLite at `.pieni/pieni.db` inside the workspace, and the latest
session for the provider/model pair is resumed on startup. Treat that file as
sensitive local data; API keys are never written to it.

## Tests

The default suite is offline and free: every provider SDK is replaced by a fake,
so no network access or API calls are needed. 128 tests:

```console
python3 -m unittest test_pieni
```

`test_sdk_wire.py` (6 tests) checks the real request and response shapes against
the installed `openai` and `openrouter` packages, using a loopback HTTP server on
127.0.0.1 instead of the providers. It needs the virtualenv on the path:

```console
PYTHONPATH=.venv/lib/python3.12/site-packages python3 -m unittest test_sdk_wire
```

It skips when the SDKs are not importable.

An optional live smoke test runs one real headless task and is skipped unless you
opt in with a provider, a model, and the matching API key:

```console
PIENI_LIVE_PROVIDER=deepseek PIENI_LIVE_MODEL=deepseek-chat \
  DEEPSEEK_API_KEY=... .venv/bin/python -m unittest test_live
```

It is not part of the default run and costs whatever your provider charges.

## Not included

Streaming output, automatic compaction, runtime provider/model switching,
reasoning-effort controls, web search, MCP, skills, plugins, multiple agent modes,
and a real OS sandbox. See `PLANS.md` for the scope and `AGENTS.md` for the rules.
