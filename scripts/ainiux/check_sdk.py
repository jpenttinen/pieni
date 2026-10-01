"""Inspect the installed provider SDKs and check what pieni.py assumes about them.

Run with the project virtualenv on the path, for example:

    PYTHONPATH=.venv/lib/python3.12/site-packages python3 scripts/ainiux/check_sdk.py

It reports, without any network access: the SDK versions, the constructor
arguments pieni passes, whether the client objects expose the methods pieni
calls, and the accepted request fields of those methods.
"""

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the project root


def show(label, value):
    print(f"{label}: {value}")


def parameters(function):
    try:
        return sorted(inspect.signature(function).parameters)
    except (TypeError, ValueError) as exc:
        return f"<signature unavailable: {exc}>"


def main():
    try:
        import openai
    except ImportError as exc:
        print(f"openai is not importable: {exc}")
        return 1
    try:
        import openrouter
    except ImportError as exc:
        print(f"openrouter is not importable: {exc}")
        return 1

    print("== versions ==")
    show("openai", getattr(openai, "__version__", "unknown"))
    show("openrouter", getattr(openrouter, "__version__", "unknown"))

    print("\n== openai ==")
    show("OpenAI accepted kwargs",
         parameters(openai.OpenAI.__init__))
    client = openai.OpenAI(api_key="not-a-real-key")
    show("has responses.create", hasattr(client.responses, "create"))
    show("responses.create kwargs", parameters(client.responses.create))
    show("has chat.completions.create", hasattr(client.chat.completions, "create"))
    show("chat.completions.create kwargs", parameters(client.chat.completions.create))
    show("base_url override accepted", "base_url" in parameters(openai.OpenAI.__init__))

    print("\n== openrouter ==")
    show("OpenRouter accepted kwargs", parameters(openrouter.OpenRouter.__init__))
    show("OpenRouter has __enter__", hasattr(openrouter.OpenRouter, "__enter__"))
    show("OpenRouter has __exit__", hasattr(openrouter.OpenRouter, "__exit__"))
    with openrouter.OpenRouter(api_key="not-a-real-key") as router:
        show("has chat.send", hasattr(router.chat, "send"))
        show("chat.send kwargs", parameters(router.chat.send))
        show("chat.send signature", str(inspect.signature(router.chat.send)))

    print("\n== pieni assumptions ==")
    import pieni  # noqa: F401  (project module, imported for its tool schemas)

    response_tool = pieni.responses_tools()[0]
    chat_tool = pieni.chat_tools()[0]
    show("responses tool keys", sorted(response_tool))
    show("chat tool shape", json.dumps({k: (sorted(v) if isinstance(v, dict) else v)
                                        for k, v in chat_tool.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
