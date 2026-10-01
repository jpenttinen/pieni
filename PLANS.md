# Pieni: a small educational coding agent

## Goal

Build a readable agent that demonstrates the basic cycle: receive a prompt,
call a model, execute requested tools, and continue until the model answers.
It should be useful to run and easy to understand, fork, and extend—not a
production agent platform.

All agent code belongs in `pieni.py`, aiming for about 750 readable lines.
Use only the standard library and the `openai` dependency, listed in
`requirements.txt`. Provide a small Bash launcher named `pieni`; tests live in
separate Python files. No agent code is to be generated during this planning task.

## First version

### Agent loop and tools

- One agent mode, shared by interactive and headless use.
- A short system prompt emphasizing KISS, YAGNI, DRY, and testing changes.
- Load `AGENTS.md` from the starting workspace root if present. No recursive
  instruction discovery in this version.
- Four tools: `read`, `write`, `edit`, and `bash`.
  - `read`: read UTF-8 text, optionally selecting a line range.
  - `write`: create or replace a UTF-8 file.
  - `edit`: replace one exact text match; fail if it is missing or ambiguous.
  - `bash`: execute a shell command from the workspace, with a timeout.
- Validate tool arguments and return concise success/error results. Bound read
  and shell output, marking truncation rather than flooding model context.
- Preserve tool-call IDs and results when continuing the model conversation.
  Do not re-execute historical tool calls when resuming a session.

### Permissions

- Default to `auto`: automatically accept shell commands unless the destructive
  command guard (DCG) flags them. Flagged commands require explicit user approval.
- Keep DCG as a small, readable regex-based function covering obvious destructive
  commands, such as recursive forced deletion and SQL `DROP TABLE`. Test both
  flagged and ordinary commands; do not attempt a complete shell/SQL parser.
- File tools automatically allow resolved paths within the starting workspace
  and the process's standard temporary directory. Outside paths require approval;
  account for `..` and symlinks when checking file-tool paths.
- Run shell commands from the workspace, but do not claim this restricts their
  filesystem access. DCG can miss destructive commands and can flag harmless
  ones; it is not a security boundary or OS sandbox.
- `yolo` skips approval and guard checks, not argument validation or timeouts.
  Make its risk explicit in help.
- In headless mode, deny any action needing approval rather than waiting for
  input. Return the denial to the model; exit nonzero if the task cannot finish.
- Approval applies only to the proposed action, not to all future actions.

### Configuration

- Use standard-library `configparser` with the default filename `pieni.ini`.
  Load optional files in this order:
  1. `~/.pieni/pieni.ini` (user settings).
  2. `pieni.ini` in the current directory when Pieni is launched (local settings).
- Merge settings per key: local values override user values; omitted local keys
  retain user values. Explicit CLI arguments override both files; built-in
  defaults apply only when neither file nor CLI supplies a value.
- Keep one `[pieni]` section with `provider`, `model`, and `permissions` settings.
  Provider/model may come from configuration instead of CLI arguments; report a
  clear error if either is still missing. Default permissions remain `auto`.
- Missing files are fine. Report unreadable files, malformed INI, and invalid
  settings clearly. Read UTF-8 and disable interpolation to keep values literal.
- API keys remain in environment variables, not configuration files. No config
  generation, alternate-file flag, or additional configuration framework yet.

### Providers and CLI

- Accept `openai`, `openrouter`, `deepseek`, or a custom base URL, with a model
  name supplied through CLI or configuration. Do not hard-code a model catalog
  or invent provider capabilities.
- Use a thin adapter: plan for OpenAI Responses and Chat Completions for the
  other endpoints, normalizing only the messages and tool calls needed here.
  Verify the relevant API details against official documentation when implementing.
- Named providers use `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `DEEPSEEK_API_KEY`.
  Custom endpoints use `OPENAI_API_KEY` if needed; do not require a key for a
  local endpoint that accepts unauthenticated requests.
- Run as `pieni PROVIDER -m MODEL` or `python3 pieni.py PROVIDER -m MODEL`.
  Provider and `-m` may be omitted when supplied by configuration.
- `-r/--run "prompt"` runs one headless task. `--permissions auto|yolo` selects
  permissions for either interface; default is `auto`.
- API keys are not command-line arguments. Use placeholders in documentation
  examples so model names do not become stale requirements.

### Interaction and history

- Start with a plain terminal prompt, not a full TUI. Print the full final answer.
  Do not depend on exposed reasoning.
- After each tool call finishes, display its name/arguments, `ok` or `error`
  (with a short cause), and elapsed time in milliseconds, for example:
  `read(path="pieni.py") -> ok, 12 ms`. Denied, failed, timed-out, and interrupted
  calls also get a result line; do not report them as successful.
- After every completed, failed, or interrupted task, in both interactive and
  headless use, display task token usage, current context usage, and elapsed time:
  `Tokens: ~2,400 | Context: ~12,000 / 1,000,000 (1.2%) | 3.1 seconds`.
  Task usage totals input and output tokens across that task's model calls;
  context usage estimates the active context, including instructions and tools,
  rather than all saved logs. Use provider usage when available and a simple,
  clearly labeled text-based estimate otherwise; no tokenizer dependency.
  On interruption, report available usage plus estimated partial work, not an
  invented exact count. Assume a 1,000,000-token context window for display only,
  not as a claim about the selected model's actual limit. Measure elapsed time
  with a monotonic clock; task duration uses one decimal place and `seconds`.
- Ctrl+C interrupts the current task and returns to the prompt; interruption at
  an idle prompt exits. Headless interruption exits nonzero. Completed actions
  are not rolled back.
- Save prompts, replies, tool calls/results, and active context in standard-library
  SQLite at `.pieni/pieni.db` under the workspace. Do not save API keys.
- On startup, restore the latest session for the selected provider and model in
  that workspace, or start fresh if none exists. Report whether a session resumed.
- Manual commands only:
  - `/compact`: ask the model for a short working summary and replace older
    conversation context only after a successful response.
  - `/compact all`: clear conversation context and start a fresh session.
  - `/permissions auto|yolo`: change permissions.
  - `/help`: concise help.
  - `/quit` and `/exit`: exit.
- Compaction retains the system prompt, workspace instructions, and tool
  definitions. Persist the new active context; keep old messages as logs, not
  automatically restored context. A failed compaction leaves context unchanged.
- Keep failures understandable: invalid configuration, network/API failures,
  unsupported tool calling, file/DB errors, and oversized context. No elaborate
  retry system or automatic context-size detection.

## Small implementation milestones

These are reviewable steps, not separate subsystems or a large PR program.

1. **Working loop:** layered INI/CLI configuration, thin provider adapter, four tools, permissions,
   and headless execution, with offline tests using mocked model responses.
2. **Interactive use:** plain prompt, interruption, tool status/timing and task
   usage/timing summaries in both interfaces, SQLite save/resume, and the small
   command set including manual compaction.
3. **Finish the example:** Bash launcher, dependency declaration, accurate README,
   and remaining failure-path tests. Optional live smoke tests with an explicitly
   selected provider/model; no paid calls in the default test run.

Test user/local/CLI configuration precedence, missing and malformed INI files,
invalid settings, successful and failing tool calls, malformed arguments, DCG approval/denial,
headless behavior, path boundaries/symlinks, empty and Unicode text, output limits,
resume without repeated tool execution, failed compaction, and tool/task summary
formatting on success, failure, and interruption (including missing provider usage).
Keep the suite comprehensive for this scope, not a separate testing framework.

## Deferred, not promised

- Advanced terminal editing, history navigation, multiline key bindings, and ESC.
- Streaming reasoning displays, exact tokenizer-based accounting, model-specific
  context-window discovery, and automatic compaction.
- Runtime provider/model switching and reasoning-effort controls.
- Web search, including provider-native search or a Tavily fallback.
- MCP, skills, plugins, multiple agent modes, and a true OS sandbox.

Revisit a deferred feature only after the small core works and its educational
value justifies the added code. There is no committed second release scope.
