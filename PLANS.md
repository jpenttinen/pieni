# GRAND PLAN

I want make minimal coding agent in Python to be used for education purposes. Rather than being complete, we should focus on it being minimal in code, yet easy to fork and extend, require fairly few dependencies.

The name for it should be "pieni". And thus run "pieni" or "python3 pieni.py"

The plan is divided in to PRs.

## PR1

The planned minimal features are:
- just one mode: no separate act/plan
- minimal destructive command guard (DCG) forbidding rm -rf, DROP table and similar using regular expressions to catch them. This could simply be a simple function which returns True/False
- in auto permission mode limit access to current directory and its subfolders and temp-directory (/tmp or C:\temp etc)
- ask permission to elevate the forbidden commands or access outside "sandbox" in auto model
- default permissions is auto, which follows minimal sandbox with DCG
- yolo permissions means no checks
- minimal system prompt, emphasizing good principles in software project: KISS, YAGNI, DRY and TDD. Make good long-term decisions when planning. You have access to ...
- load AGENTS.md if present
- agentic tools: read, write, edit and bash 
- supporting OpenAI compatible endpoints: OpenAI, OpenRouter and DeepSeek. And their corresponding API keys: OPENAI_API_KEY, OPENROUTER_API_KEY and DEEPSEEK_API_KEY, assuming they are set as environment variables as the user. Also support custom base_url and secret key e.g. for local use http://localhost:30000, secret: 123
- minimal thin adapter for different provider. OpenAI uses Responses API by default, others use Chat Completions API
- minimal tui with up/down arrow to browse history, enter to send and alt+enter/shift+enter to enter newline:
> user prompt
Thinking: Show first 120 characters and accumulate more with "."
# Show tool calls like this read(path="actual path", start_line...) -> ok/error (cause), time elapsed in ms
Show full response text
- ESC or ctlr+C interrupts task, but doesn't exit agent.
- after each completed task display estimated tokens and how much in total context, elapsed time in seconds e.g. 6.5 seconds
- graceful error handling
- automatically store and upon start load the log of prompts & responses etc. in Sqlite3 DB in .pieni/pieni.db
- no support for MCP, SKILLS
- have the following CLI:
pieni provider -m "model" # start agent with provider and mode name
pieni deepseek -m "deepseek-flash"
pieni openai -m "gpt-6-luna"
pieni openrouter -m "glm-5.3-flash"
pieni http://localhost:30000 -m "qwen3.6-35b-a3b"
headless mode -r/--run "prompt"
- Have the following commands:
/compact {all] # compact context, 75% auto compact limit, /compact all removes everything from context.
/permissions auto|yolo
/provider [provider-name]
/model [model-name]
/reasoning [reasoning-effort]
/help # very concise help
/quit # quit
/exit # same

Aim to spend max. 750 lines for this. Target is just one Python file pieni.py (you must create it). Generate also requirements.txt.

Generate comprehensive test suite: unit tests, mock up test, smoke test and integration tests to different Python files.

There is a local Qwen3.6-35A-A3B model at http://localhost:30000, which you can use for testing for free. Or you can use Deepseek as provider and "deepseek-flash" as the model or OpenAI and gpt-6-luna as the model.

---

## PR2:

Add web_search tool. Use built-in web_search for the popular models and use Tavily as backup choice if the provider has no web_search. TAVILY_API_KEY has been set.
