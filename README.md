# Studio Assistant

A chat window that drives your creative apps with a **local LLM**. One tab per app —
**After Effects** and **DaVinci Resolve** today.

Ask in plain language — *"make a 1920x1080 title card, 5 seconds at 24fps"* — and it
builds it in the app the current tab points at, one real undo step at a time.

Inference runs on a machine on the tailnet. Your project data never leaves the LAN,
and the GPU in the workstation stays free for rendering.

## Running it

Double-click **Studio Assistant** on the Desktop or in the Start Menu. No terminal.

Pick the app you want to talk to from the tab strip. Each tab is a separate
conversation with its own tools — After Effects never sees Resolve's history, and
neither app's tools are offered to the other model turn. `Ctrl+Tab` cycles tabs.

Bridges start **lazily**: a tab connects the first time you open it, then warms the
model against that app's own tool schemas so your first question there comes back in
seconds rather than a minute. A session that only touches After Effects never spawns
Resolve's server. If the app itself isn't running, a **Start <app>** button appears in
the header — one click launches it and waits.

The left rail shows what's installed on this machine and what the agent can currently
reach, with live status dots per app.

### From a terminal

```bash
python studio_chat.py                                     # the GUI
python studio_agent.py "what comps are in this project"   # one-shot, After Effects
python studio_agent.py --app resolve "what's on the timeline"
python studio_agent.py --app resolve                      # interactive REPL
python studio_agent.py --list-groups                      # tool families, per app
python studio_agent.py --app resolve --list-tools
```

Useful flags: `--app`, `--groups` (which tool families to expose), `--all-tools`,
`--model`, `--host`, `--max-steps`. Environment overrides: `STUDIO_HOST`,
`STUDIO_MODEL`, `RESOLVE_MCP_DIR`.

## The apps it drives

| App | Bridge | Needs |
|---|---|---|
| After Effects | `@engine-room/after-effects-mcp` over `npx`, talking to the CEP panel on `127.0.0.1:7777` | Node / `npx` on PATH, and the panel installed (`setup_panel`, with AE closed) |
| DaVinci Resolve | `davinci-resolve-mcp` from `~/davinci-resolve-mcp` | Resolve Studio, with *External scripting using* set to **Local** |

Each tab carries its own system prompt — a briefing on that bridge's ids, units and
conventions, because the local model knows the app in general but has never seen this
bridge. It is what keeps After Effects colours in 0..1 and Resolve's track numbering
off by the right one.

Adding a third app is a registry entry in `studio_agent.py`, not a code change —
`AGENTS.md` says what an entry has to supply, the briefing included.

## Requirements

- Windows, with at least one of the apps above.
- **Python 3.9+** with Tkinter — the standard python.org build is fine.
- An OpenAI-compatible endpoint reachable on the network. Built and tested against
  **LM Studio**; the default model is `qwen3-coder-30b-a3b-instruct`, but whatever is
  loaded gets picked automatically.

No `pip install` — the whole thing is standard library, deliberately.

## Layout

| Path | What |
|---|---|
| `studio_chat.py` | The Tkinter GUI: tab strip, one `Session` per app |
| `studio_agent.py` | Engine: app registry, MCP client, LLM client, schema sanitizing, probes. Also a CLI |
| `Studio Assistant.cmd` | Console-free launcher used by the shortcuts |
| `make_icon.py` | Regenerates `studio-assistant.ico`, the shortcut and taskbar mark |
| `tests/` | Offline tests — no network, no creative apps |
| `AGENTS.md` | Notes for anyone changing the code. **Read this first** |

## Contributing

Run `python -m unittest discover -s tests` before and after a change. `AGENTS.md`
documents several non-obvious constraints — a stdlib-only rule, two JSON/schema
incompatibilities that fail loudly *and* quietly, two Tkinter traps, a warm-up that
looks removable and isn't, and what each app's system prompt has to tell the model.
