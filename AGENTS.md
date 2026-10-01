# Pieni development instructions

## Purpose and scope

Pieni is a tiny Python coding agent for learning how agents work. Favor a clear,
working example over a complete product. Follow the scope and milestones in
`PLANS.md`; do not add features just because larger agents have them.

## Keep it small

- Keep all agent implementation in `pieni.py`, with a Bash launcher named `pieni`.
- Use the Python standard library and only one external dependency: `openai`.
- Use `configparser` for INI settings: user file first, launch-directory file
  second, and explicit CLI arguments last. Keep configuration small.
- Aim for about 750 lines of agent code. This is a guideline, not a reason to
  compress code into unreadable one-liners or omit necessary checks.
- Prefer straightforward functions and a few clearly separated sections for
  configuration, tools, permissions, persistence, and the agent loop.
- Apply KISS, YAGNI, and DRY. Avoid frameworks, plugins, elaborate abstractions,
  and unrelated changes.
- Use short comments to explain non-obvious decisions, especially in the agent
  loop and permission checks. Keep examples and documentation honest.

## Behavior and safety

- Keep one agent mode; permissions are a separate `auto`/`yolo` setting.
- Treat the destructive-command guard as best-effort checking, not a secure
  sandbox. Never describe arbitrary shell execution as safely contained.
- Keep API keys in environment variables; do not put them in source, examples,
  or logs. Treat saved conversations as potentially sensitive local data.
- Handle invalid inputs, tool failures, provider errors, and interruption clearly.
  Do not silently retry actions that may already have changed files.

## Tests and changes

- Keep tests outside `pieni.py`; use standard-library `unittest` and mocks.
- Test the core behavior and failure paths, including permissions, paths,
  Unicode, persistence, and the tool-call loop.
- Keep default tests offline and free of API charges. Live integration tests
  must be opt-in and use provider/model settings supplied by the developer.
- Run relevant offline tests after implementation changes; report what was
  tested and what was not verified.
- When a task is documentation or planning only, do not generate agent code.
