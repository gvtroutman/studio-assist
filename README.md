# Studio Assistant

A chat window that drives your creative apps with a **local LLM**. One tab per app —
**After Effects**, **Premiere Pro**, **Photoshop**, **Illustrator**, **DaVinci Resolve**,
**ComfyUI** and **OpenCode** today, plus **any app you have an MCP bridge for** — and a
**Chat** tab for everything that needs no app at all — it reads your files and the web
instead.

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

**Chat** is the same window with no creative app behind it, for the questions between
the work — what frame rate to finish in, how long 240 frames runs at 23.976, talking
an approach through before you build it. What it has instead are read-only tools for
looking things up: it can list and search folders on this PC, read a brief, a script or
a `.docx`, search the web and read a page, and it cites what it read. It writes
nothing, refuses files that hold credentials, starts instantly, needs nothing
installed, and it will tell you to use an app tab rather than pretend it changed your
project.

Bridges start **lazily**: a tab connects the first time you open it, then warms the
model against that app's own tool schemas so your first question there comes back in
seconds rather than a minute. A session that only touches After Effects never spawns
Resolve's server. If the app itself isn't running, a **Start <app>** button appears in
the header — one click launches it and waits. ComfyUI is the exception: it runs on the
LLM PC, so its button is **Check ComfyUI** and it tells you where to start it.
**Start OpenCode** starts a Docker container rather than a program — the first time it
also builds the image, which takes a few minutes.

The left rail shows what's installed on **this machine** — named, so you know which
one — with each app's own icon and a live status dot for the ones the agent can
drive. It's your list, not the machine's: **pin** the apps you work in to the top,
**×** the ones you haven't set up out of the way, and **+** in the heading brings any
of them back. Click a drivable app to open its tab.

Underneath, **Connections** is two rows. *Inference* is the model host; its menu has
one thing in it, **Connect**. *Bridges* opens a menu of every bridge that exists and
what it can currently reach; pick one to see every tool it offers, grouped, and which
of them this chat puts in front of the model.

**When the model host is not there.** The window probes it as it opens, with a short
timeout, so a PC still waking, a tailnet still coming up or an LM Studio server not yet
started all read as *no inference host — trying again*. Nothing is lost, and nothing
needs pressing: the window probes again every 30 seconds, quietly, and the tab you are
looking at starts the moment the host answers; the others start when you switch to
them. **Connect** — the header button that appears in place of *Start*, the Inference
row, or **Bridges ▸ Connect to the inference host** — tries now instead of in 30
seconds. The same happens if the host drops away in the middle of a conversation (the
Inference row goes red and says *unreachable*); sending again would do just as well,
since every request is a fresh connection.

The transcript says *which* half is missing, because the probe alone cannot tell:
Windows drops a port with no listener rather than refusing it, so a PC that is asleep
and a PC that is up with LM Studio's server stopped both look like a timeout. The
window asks the tailnet (`tailscale ping`) and then says either *the LLM PC is not
answering at all* — it is asleep, off, or off the tailnet — or *the LLM PC is up, but
LM Studio's server is not answering*. A locked screen does not stop LM Studio; sleep
does, and a restart brings the PC back without it. On the LLM PC, the lasting fix is
`powercfg /change standby-timeout-ac 0` (lock all you like, just don't sleep) and LM
Studio's *Enable Local LLM Service on Login* (Settings ▸ Developer), so the server is
there after any restart with nobody signed in.

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
python studio_agent.py --app photoshop "what layers are in this document"
python studio_agent.py --mcp "npx -y some-mcp-server" --name Blender "what's in the scene"
```

Useful flags: `--app`, `--groups` (which tool families to expose), `--all-tools`,
`--model`, `--draft` (the speculative-decoding draft model, or `off`), `--host`,
`--max-steps`, and `--mcp "<command line>"` to drive any MCP
stdio bridge that is not in the registry (`--name` says what to call the app). Environment overrides: `STUDIO_HOST`,
`STUDIO_MODEL`, `STUDIO_MODEL_<APP>` (one app's model, e.g. `STUDIO_MODEL_COMFYUI`),
`STUDIO_DRAFT_MODEL`, `STUDIO_VISION_MODEL`,
`RESOLVE_MCP_DIR`, `COMFYUI_URL`, `COMFYUI_OUTPUT_DIR`.

**Speculative decoding is on when the host can pair the model.** A small model of
the same family — `qwen3-0.6b` beside `qwen3-coder-30b-a3b-instruct`,
`qwen2.5-coder-0.5b-instruct` beside `qwen2.5-coder-14b-instruct` — drafts the next
few tokens and the big model checks them in one pass, so answers read the same and
arrive sooner. Studio Assistant asks LM Studio for it on every request
(`draft_model`), and LM Studio loads the draft the first time it is asked for, so
download one in LM Studio and it is used from the next window. The
Inference row names it (`draft: qwen3-0.6b`) and, once LM Studio reports how the
draft is doing, the share of its guesses the model kept — a draft that is mostly
wrong is slower than none, and that number says so. `STUDIO_DRAFT_MODEL` pins a
different one (the two must share a vocabulary — LM Studio refuses a mismatch, and
the tab says so once and carries on without) or `off` turns it off. Newer LM Studio
takes the draft only when the model is loaded, not per request; it refuses the
request, the tab says so once, and runs without — `off` skips the asking.

**The model is loaded with room to work.** Every tab's briefing and tool schemas
take 8,000–20,000 tokens before you have typed anything, and LM Studio loads a model
with an 8,192-token window unless told otherwise — under that the model loses its own
tool results and asks the same question again, or is cut off mid-reply. So when a tab
starts, Studio Assistant loads the model itself (or unloads and reloads it) with a
window that holds the tab's briefing and a conversation — 16,384 or 32,768, never
more than the model supports — and the tab says so in one line. Nothing to set on the
LLM PC. A tab that starts while another is mid-request leaves the model alone and
prints the `lms load … --context-length` to run by hand instead.

**Which model.** `STUDIO_MODEL` if you set it; else `qwen3-coder-30b-a3b-instruct`
when the host has it in VRAM; else whatever else you have loaded in LM Studio, apart
from the vision model and the draft the app itself asks for; else the first known-good
model the host has downloaded. Every tab, ComfyUI's included, uses that one model —
the tab used to prefer a 1.7B so the GPU stayed free for the pictures, and the 1.7B
talked about generating instead of doing it. `STUDIO_MODEL_COMFYUI` pins a smaller one
on a host that cannot hold the diffusion model and the 30B together.

## The apps it drives

| App | Bridge | Needs |
|---|---|---|
| After Effects | `@engine-room/after-effects-mcp` over `npx`, talking to the CEP panel on `127.0.0.1:7777` | Node / `npx` on PATH, and the panel installed (`setup_panel`, with AE closed) |
| Premiere Pro (Beta) | `studio_premiere_mcp.py` (in this folder), posting ExtendScript to the **Studio Assistant Bridge** panel inside Premiere on `127.0.0.1:7787` (`STUDIO_PREMIERE_PORT`) | The panel installed: `python studio_premiere_mcp.py --install-panel` with Premiere closed, then open it once from *Window > Extensions*. CEP's `PlayerDebugMode` must be on (the installer says if it is not) |
| Photoshop | `studio_photoshop_mcp.py` (in this folder), running ExtendScript inside Photoshop through its Windows COM automation (`Photoshop.Application`) | Photoshop installed. Nothing to install inside it — no panel, no plugin |
| Illustrator | `studio_illustrator_mcp.py` (in this folder), the same way through `Illustrator.Application` | Illustrator installed. Nothing to install inside it |
| DaVinci Resolve | `davinci-resolve-mcp` from `~/davinci-resolve-mcp` | Resolve Studio, with *External scripting using* set to **Local** |
| ComfyUI | `studio_comfy_mcp.py` (in this folder), talking HTTP to ComfyUI on the LLM PC — `http://100.127.17.38:8188` unless `COMFYUI_URL` says otherwise | ComfyUI started on that machine with `--listen` (so it accepts connections from the workstation), and at least one checkpoint installed there |
| OpenCode | `studio_opencode_mcp.py` (in this folder), talking HTTP to `opencode serve` inside a Docker container on `127.0.0.1:4096` (`OPENCODE_URL`) | Docker Desktop installed and running. The image is built from `opencode/Dockerfile` on first start |

Photoshop and Illustrator need no bridge installed anywhere: on Windows both register
COM automation, and its one method that matters runs ExtendScript inside the live app.
The bridge keeps a PowerShell worker holding that handle, so a call costs about half a
second. If the app is closed, the first call starts it (COM does that), which takes a
while; `ps_status` / `ai_status` say whether it is running without starting it. Layers
are addressed by Photoshop's own `layer_id`, Illustrator items by their `uuid`, and
both tabs speak pixels/points from the top-left with y down — Illustrator itself counts
y upward; the bridge flips it so the model has one convention across every app.
`ps_screenshot` and `ai_screenshot` put a picture in the transcript; `ps_place_file`
and `ai_place_file` bring a ComfyUI picture in.

Premiere Pro has no COM automation, so its bridge is a small CEP panel (`premiere_panel/`)
that runs a loopback web server inside Premiere and evaluates the scripts the bridge
sends it — the same road After Effects' bridge takes. The Beta is looked for first, since
that is the Premiere this studio cuts in. Timeline clips are addressed by `clip_id` and
project items by `item_id` (Premiere's own stable ids), time is seconds everywhere and
tracks count from 1. `ppro_screenshot` exports a frame into the transcript,
`ppro_import_files` brings footage or a ComfyUI picture in, and `ppro_export` renders
with any installed preset — `ppro_list_presets` shows them. If a tool says it cannot
reach Premiere, `ppro_status` says the one thing to do.

**Any other app — Audition, Blender, a DAW — with a bridge you have installed:** right-click its row in the sidebar (or *File → Connect an MCP bridge…*)
and enter the command line that starts the bridge, the same one its README puts in an
MCP client config. The tab opens, reads the bridge's tools and its own instructions,
groups the tools by name prefix so a large bridge can be narrowed with *Choose
capabilities*, and briefs the model from what the bridge said about itself. The entry
is saved in settings; right-click the row again to edit or forget it. Premiere Pro has
several community bridges (each installs a CEP panel inside Premiere); the dialog
names the ones seen.

ComfyUI is the one app that runs on the *other* machine: its GPU generates the
images so the workstation's stays free for rendering. The bridge still runs here,
and every picture it makes is copied back to `~/Pictures/ComfyUI` (or
`COMFYUI_OUTPUT_DIR`) and shown in the transcript, so the After Effects and Resolve
tabs can import it by path. Ask for a checkpoint list first — the tab is briefed on
which sizes and step counts suit SDXL, SD 1.5 and turbo models, but it has to know
which one it is driving.

OpenCode is a coding agent — it writes and runs code on its own — so it never runs on
this PC's own filesystem. **It lives in a container that can see exactly one folder**:
the workspace, `%LOCALAPPDATA%\StudioAssistant\opencode-workspace` (or
`OPENCODE_WORKSPACE`). Drop files there, or let the tab put them there, and ask; what
it builds lands in that folder and nowhere else. Its port is published to loopback
only, it runs as an unprivileged user with capabilities dropped, and it cannot reach
After Effects, Resolve or their projects. It uses the same LM Studio host as the other
tabs (`STUDIO_MODEL_OPENCODE` pins its model). The window does not stop the container
when it closes; `docker stop studio-opencode` does, and the workspace stays.

Each tab carries its own system prompt — a briefing on that bridge's ids, units and
conventions, because the local model knows the app in general but has never seen this
bridge. It is what keeps After Effects colours in 0..1 and Resolve's track numbering
off by the right one.

After Effects exposes shape contents, path and property editing, masks, and text
animators by default. Try *"Add a centred red circle, 300 pixels across, to my
comp."* The shape briefing covers geometry plus fill/stroke, grouping, and comp
coordinates; creating an empty shape layer alone is not a completed drawing.
Restart Studio Assistant after updating so its tabs load the new tools and prompt.

Adding an app for good is a registry entry in `studio_agent.py`, not a code change —
`AGENTS.md` says what an entry has to supply, the briefing included. Connecting one
for now is the dialog above.

## Requirements

- Windows, with at least one of the apps above.
- **Python 3.9+** with Tkinter — the standard python.org build is fine.
- An OpenAI-compatible endpoint reachable on the network. Built and tested against
  **LM Studio**; the default model is `qwen3-coder-30b-a3b-instruct`, and a model you
  load by hand is used when that one is not loaded (see *Which model* above). The
  window loads and reloads models on the host itself, so LM Studio's server has to be
  the one with the REST API (`/api/v1`), which any recent build is.

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

Pictures go the other way too: the picture button beside the input (or Ctrl+O)
attaches files to the next message, in every tab. The model is given each file's
name, dimensions and path, so "place this in my comp" or "turn this sketch into a
painted version" can hand the file to the app's own import tool; the picture is shown
in the transcript under your message. What the picture *looks like* reaches the
model through the **vision model**: it describes each picture and that description
goes into the brief. OpenCode sees only its workspace, so pictures attached there
are copied into `attachments/` inside it.

The executing model reads text, and every app answers a screenshot with a picture,
so a vision model is part of every tab, not an option. At start-up Studio Assistant
picks one from what the **same remote inference host** serves: `STUDIO_VISION_MODEL`
if you pin one and it is served, else the executing model itself when it can see,
else one already in VRAM, else a known-good vision model (Qwen-VL, Gemma), else any
the host has downloaded. It need not be loaded: Studio Assistant asks LM Studio to
load it while the tabs start (the Inference row says `loading …`, then `sees: …`),
and a host that cannot be asked loads it on first use. Returned frames and the task
brief go to it; it says what the frame actually shows and how it falls short, and
that comes back to the executing model for the next step. No model is loaded on the
workstation. Only if the host has no vision model downloaded at all does the row
turn amber; every tab says so once, and the model works blind: attached pictures are
names and paths to it, and previews reach only you — download one in LM Studio and
press **Connect**. Still frames cannot verify motion or audio.

## What it knows, learns and asks

The model behind every tab is small and local. Four things stand between it and
guessing:

**It looks things up.** Every app tab carries the Chat tab's read-only tools beside
its own bridge: list, search and read files on this PC, search the web, read a page
as text. The prompt tells it to read the brief rather than imagine it, to look a
feature or an expression up before guessing, and to say which page it relied on;
each app's entry names its own documentation (the After Effects scripting guide and
expression reference, the Premiere and Illustrator scripting guides, Resolve's
scripting API reference, ComfyUI's docs and examples, Photoshop's scripting
reference). Adobe's `helpx.adobe.com` refuses direct reads, so it is reached through
search results only. Reading a page never counts as checking that an edit landed.

**It learns.** Each app has a notebook of one-line lessons, kept beside the settings
in `lessons/<app>.json` and folded into the app's briefing at the start of every
session. Lessons arrive four ways: a call the validator refused (an invented action,
a misspelled key) is learned as it stands; a message of yours that begins "remember"
or "from now on" is kept in your words; the model keeps one itself with
`studio_remember` when you correct it or tell it how you work; and after a run that
had an error in it, or a brief that read as a correction ("no, I meant…"), one extra
short request asks the model what it should remember — a clean run costs nothing.
Lessons kept mid-session ride at the tail of the request until the next New chat.
**File → Lessons for this tab** lists them, with a **forget** for the ones that
turned out wrong.

**It knows the studio.** **File → About this studio** opens a short Markdown brief —
who you are, what the studio makes, brands and house styles, deliverables and
formats, how projects and timelines are organised, what to ask before doing. Saved,
it is the last part of every tab's briefing (a tab whose conversation has not started
takes it at once; the rest on their next New chat). Nothing of the template reaches
the model until you have written over it. Beside it every tab carries the craft of
its kind of work — how an edit is cut (read the whole timeline first, plan the
assembly as a list, handles, J- and L-cuts, track discipline, durations in frames),
how motion and design work is built, how images are prompted — and a rule set for
open briefs: name directions, choose, say why, build the simplest version, look,
refine.

**It asks properly.** When an answer changes what it would build and cannot be read
from the project — which frame rate, which take, which platforms — the model calls
`studio_ask`, and the question appears in the transcript as a form: a button per
choice with its description, tick-boxes and **Send these** when several may apply,
and **Something else…** to type your own. A click is your next message; the form
then greys out. On the CLI the same question is a numbered list.

## Layout

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
| `studio_toolsmith.py` | Tools the model makes for itself: named sequences of the tools it already has |
| `studio_lessons.py` | What the model learns per app: the notebook, `studio_remember`, the end-of-task reflection |
| `studio_agent.py` | Engine: app registry, MCP client, LLM client, schema sanitizing, probes. Also a CLI |
| `studio_mcp.py` | The MCP harness: the server our bridges run on, an in-process client, and `check` — holds any bridge to what the executor and the model need |
| `studio_icons.py` | Reads an app's icon out of its own `.exe`, and writes PNGs. No dependencies, nothing shipped |
| `Studio Assistant.cmd` | Console-free launcher used by the shortcuts |
| `make_icon.py` | Regenerates `studio-assistant.ico`, the shortcut and taskbar mark |
| `tests/` | Offline tests — no network, no creative apps. `tests/contracts/` records what the installed bridges expose, so the registry is checked against the real thing |
| `AGENTS.md` | Notes for anyone changing the code. **Read this first** |

## Checking a bridge

Every bridge - the four written here and the two installed ones - can be held to what
the executor and the model actually need, without opening the app:

```bash
python studio_mcp.py check --app after-effects
```

It starts the bridge, negotiates the protocol, lists the tools, and reports what will
go wrong before a model finds out: schemas the inference host would refuse, tools no
group exposes, a prompt that teaches a tool the tab was never given, a contract too big
to leave room for conversation. `--call` also runs the bridge's harmless reads.
`snapshot` records the contract into `tests/contracts/`, and the tests then hold the
registry to it offline.

## Contributing

Run `python -m unittest discover -s tests` before and after a change. `AGENTS.md`
documents several non-obvious constraints — a stdlib-only rule, two JSON/schema
incompatibilities that fail loudly *and* quietly, two Tkinter traps, a warm-up that
looks removable and isn't, and what each app's system prompt has to tell the model.
