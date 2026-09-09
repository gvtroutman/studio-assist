# AE Agent

A chat window that drives **Adobe After Effects** with a **local LLM**.

Ask for it in plain language — *"make a 1920x1080 title card, 5 seconds at 24fps"* —
and it builds it in your open After Effects project, one real undo step at a time.

Inference runs on a machine on the tailnet. Your project data never leaves the LAN,
and the GPU in the workstation stays free for rendering.

![sidebar shows detected apps and live connections; chat on the right]

## Running it

Double-click **AE Agent** on the Desktop or in the Start Menu. No terminal.

On launch it checks the inference host, picks a model, starts the After Effects
bridge, and warms the model so your first question comes back in seconds rather than
a minute. If After Effects isn't running, a **Start After Effects** button appears in
the header — one click launches it and waits for the bridge.

The left rail shows what's installed on the machine and what the agent can currently
reach, with live status dots.

### From a terminal

```bash
python ae_chat.py                                   # the GUI
python ae_agent.py "what comps are in this project" # one-shot CLI
python ae_agent.py                                  # interactive REPL
python ae_agent.py --list-tools                     # what's exposed
```

Useful flags: `--groups` (which tool families to expose), `--all-tools`, `--model`,
`--host`, `--max-steps`. Environment overrides: `AE_AGENT_HOST`, `AE_AGENT_MODEL`.

## Requirements

- Windows with **After Effects** and the `@engine-room/after-effects-mcp` CEP panel
  installed (`setup_panel` installs it; After Effects must be closed at the time).
- **Python 3.9+** with Tkinter — the standard python.org build is fine.
- **Node / npx** on PATH, for the bridge.
- An OpenAI-compatible endpoint reachable on the network. Built and tested against
  **LM Studio**; the default model is `qwen3-coder-30b-a3b-instruct`, but whatever is
  loaded gets picked automatically.

No `pip install` — the whole thing is standard library, deliberately.

## Layout

| Path | What |
|---|---|
| `ae_chat.py` | The Tkinter GUI |
| `ae_agent.py` | Engine: MCP client, LLM client, schema sanitizing, probes. Also a CLI |
| `AE Agent.cmd` | Console-free launcher used by the shortcuts |
| `tests/` | Offline tests — no network, no After Effects |
| `AGENTS.md` | Notes for anyone changing the code. **Read this first** |

## Contributing

Run `python -m unittest discover -s tests` before and after a change. `AGENTS.md`
documents several non-obvious constraints — a stdlib-only rule, a JSON-Schema
incompatibility, two Tkinter traps, and a warm-up that looks removable and isn't.
