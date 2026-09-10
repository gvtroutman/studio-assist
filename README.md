# Studio Assistant

A chat window that drives your creative apps with a **local LLM**. One tab per app —
**After Effects** and **DaVinci Resolve** today — and a **Chat** tab for everything
that needs no app at all.

Ask in plain language — *"make a 1920x1080 title card, 5 seconds at 24fps"* — and it
builds it in the app the current tab points at, one real undo step at a time.

Inference runs on a machine on the tailnet. Your project data never leaves the LAN,
and the GPU in the workstation stays free for rendering.

## Running it

Double-click **Studio Assistant** on the Desktop or in the Start Menu. No terminal.

Pick the app you want to talk to from the tab strip. Each tab is a separate
conversation with its own tools — After Effects never sees Resolve's history, and
neither app's tools are offered to the other model turn. `Ctrl+Tab` cycles tabs,
`+` opens one for another app and `×` closes one; whatever is open when you quit is
what comes back next time.

**Chat** is the same window with the bridge and the tools left out: the model on its
own, for the questions between the work — what frame rate to finish in, how long 240
frames runs at 23.976, talking an approach through before you build it. It starts
instantly, needs nothing installed, and it will tell you to use an app tab rather
than pretend it changed your project.

Bridges start **lazily**: a tab connects the first time you open it, then warms the
model against that app's own tool schemas so your first question there comes back in
seconds rather than a minute. A session that only touches After Effects never spawns
Resolve's server. If the app itself isn't running, a **Start <app>** button appears in
the header — one click launches it and waits.

The left rail shows what's installed on **this machine** — named, so you know which
one — with each app's own icon and a live status dot for the ones the agent can
drive. It's your list, not the machine's: **pin** the apps you work in to the top,
**×** the ones you haven't set up out of the way, and **+** in the heading brings any
of them back. Click a drivable app to open its tab.

Underneath, **Connections** is two rows. *Inference* is the model host. *Bridges*
opens a menu of every bridge that exists and what it can currently reach; pick one to
see every tool it offers, grouped, and which of them this chat puts in front of the
model.

The menu bar carries the rest: **File** for chats and tabs, **View** to switch
between dark and light, **Bridges** to jump straight to a tool list.
**File ▸ Preferences** (`Ctrl+,`) opens the appearance switch and restores hidden
apps. Preferences, pinned and hidden apps and open tabs are remembered in
`%APPDATA%\StudioAssistant\settings.json`.

### From a terminal

```bash
python studio_chat.py                                     # the GUI
python studio_agent.py "what comps are in this project"   # one-shot, After Effects
python studio_agent.py --app resolve "what's on the timeline"
python studio_agent.py --app resolve                      # interactive REPL
python studio_agent.py --app chat "how long is 240 frames at 23.976"
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

After Effects exposes shape contents, path and property editing, masks, and text
animators by default. Try *"Add a centred red circle, 300 pixels across, to my
comp."* The shape briefing covers geometry plus fill/stroke, grouping, and comp
coordinates; creating an empty shape layer alone is not a completed drawing.
Restart Studio Assistant after updating so its tabs load the new tools and prompt.

Adding a third app is a registry entry in `studio_agent.py`, not a code change —
`AGENTS.md` says what an entry has to supply, the briefing included.

## Requirements

- Windows, with at least one of the apps above.
- **Python 3.9+** with Tkinter — the standard python.org build is fine.
- An OpenAI-compatible endpoint reachable on the network. Built and tested against
  **LM Studio**; the default model is `qwen3-coder-30b-a3b-instruct`, but whatever is
  loaded gets picked automatically.

No `pip install` — the whole thing is standard library, deliberately.

## Completing and recovering tasks

The GUI and CLI use the same executor. It validates tool names and arguments,
checks documented Resolve actions, blocks Resolve's `quit` action, and stops
repeated failures. A write followed by a final answer triggers a request to read
back the changed target. This catches missing verification steps; it does not
independently prove that the result is correct or visually polished.

For substantial work, the agent can maintain a brief, plan, object IDs, acceptance
checks and unresolved issues through its internal task-record tool. Built-in
guidance covers animation, timeline assembly and delivery. Full conversation
history stays in the saved task; inference receives a bounded selection of whole
tool exchanges plus the brief and task record. If the fixed context is too large,
execution stops with an explanation instead of silently clipping the contract.

- **Stop:** while a task runs, Send becomes Stop. It prevents subsequent tool
  dispatches. The current network request or app operation may need to return or
  time out first; completed edits remain in the creative app.
- **File → Task progress:** inspect the saved brief, plan, checks and issues.
- **File → Resume saved task:** reopen an earlier task for the active app, then
  send a message to continue. The agent must inspect the current project first.
- **Bridges → Choose capabilities for current tab:** enable additional groups,
  such as AE asset import or Resolve Fusion. Default groups remain enabled because
  the app's prompt depends on them. Applying changes warms the selected tool set.

GUI tasks are saved atomically before tool dispatch and after tool results under
`%APPDATA%\StudioAssistant\tasks\<app>\<task-id>.json`. With `STUDIO_SETTINGS`, the
tasks folder sits beside that settings file. These files include prompts, arguments
and complete textual tool results. New chat starts a new record; old task files
remain available. Restoring a task restores conversational context, **not** the
creative application's project or an undo state. CLI invocations use the same
validation and verification loop but do not persist task files.

When a write's result is lost, its outcome is recorded as unknown and that exact
call cannot be repeated in the same saved task. Inspect the project and reconcile
the result before proceeding. If task saving fails, execution stops. Task history
does not replace a project backup; backup/duplication guidance uses only tools the
bridge actually provides.

AE inspection tools are enabled by default. Image results are displayed inline
when Tk supports their format (PNG/GIF); previews are not restored from task files.
For optional visual critique, set `STUDIO_VISION_MODEL` to the ID of a vision-capable
model served by the **same remote inference host**, then restart Studio Assistant.
Returned image frames and the task brief go to that model; critique comes back to
the executing model for refinement. No model is loaded on the workstation. Without
that setting, the user can review previews and the text model is told that it has
not assessed their visual quality. Still frames cannot verify motion or audio.

## Layout

### Workflow layer (first increment)

The GUI and CLI expose `studio_workflow_capabilities`, reporting availability and
specific blockers for the thirteen planned production workflows. Currently
`inspect_project`, `inspect_comp`, and `inspect_layer` adapt the enabled AE bridge
tools `get_project_summary`, `get_comp`, and `get_layer_full`. They inherit the
bridge's argument schemas and full descriptions; comp inspection returns comp
metadata, not a recursive inspection of all its layers. These adapters use the
shared executor's original-schema validation, journal, recovery and preview path.
They are exposed only when their underlying tool is enabled. Resolve adapters
are not implemented yet.

Audio inspection, transcripts, media matching/placement, Miter and Ellwood styling,
red-card construction, speech synchronization, and verification evaluators are
explicitly unavailable in this first increment. They are not placeholder callable
tools. The capability report explains each missing integration. A style/template
reference for each brand and a timestamped transcript provider are needed before those workflows
can be implemented accurately. Inspection results are observations, not automatic
visual or timing verification. Inference remains on the remote host.

Styling has two separate workflows: `apply_miter_style` and
`apply_ellwood_style`, each with its own brand reference and implementation.

### Tools the model makes for itself

When the same run of calls keeps coming up, the model can name it: `studio_tool_create`
records a fixed sequence of the tools this tab already has, with its own inputs filled
into their arguments, and offers it back as one tool from the next message on. A made
tool is data, never code — there is no eval anywhere in this — and every step is
dispatched through the ordinary executor, so it cannot reach a tool the tab was not
given, cannot skip validation against the bridge's own schema or the Resolve
prohibition, and cannot keep its work out of the task journal. Made tools are saved per
app beside the settings file, so they are there in the next session and on the CLI too,
and the app's tools window lists them with a **forget** button for the ones that turned
out badly.

| Path | What |
|---|---|
| `studio_chat.py` | The Tkinter GUI: tab strip, one `Session` per app, plus the app-less Chat tab |
| `studio_tasks.py` | Shared executor, argument checks, task records, context budgeting and recovery |
| `studio_workflows.py` | Capability catalog and schema-backed inspection adapters |
| `studio_toolsmith.py` | Tools the model makes for itself: named sequences of the tools it already has |
| `studio_agent.py` | Engine: app registry, MCP client, LLM client, schema sanitizing, probes. Also a CLI |
| `studio_icons.py` | Reads an app's icon out of its own `.exe`, and writes PNGs. No dependencies, nothing shipped |
| `Studio Assistant.cmd` | Console-free launcher used by the shortcuts |
| `make_icon.py` | Regenerates `studio-assistant.ico`, the shortcut and taskbar mark |
| `tests/` | Offline tests — no network, no creative apps |
| `AGENTS.md` | Notes for anyone changing the code. **Read this first** |

## Contributing

Run `python -m unittest discover -s tests` before and after a change. `AGENTS.md`
documents several non-obvious constraints — a stdlib-only rule, two JSON/schema
incompatibilities that fail loudly *and* quietly, two Tkinter traps, a warm-up that
looks removable and isn't, and what each app's system prompt has to tell the model.
