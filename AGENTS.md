# AGENTS.md

Working notes for anyone — human or agent — changing this project.

## What this is

A chat app that drives creative apps with a **local LLM**, one tab per app — plus
a **Chat** tab with no app behind it, for the questions that need no app: it reads
this PC's files and the web instead. Two moving parts:

- **`studio_agent.py`** — the engine. The **app registry**, an MCP stdio client, an
  OpenAI-compatible LLM client (streaming and not), JSON-Schema sanitizing, and
  environment probes. Also a working CLI: `python studio_agent.py --app resolve
  "what's on the timeline"`, or bare for a REPL.
- **`studio_chat.py`** — the Tkinter GUI, and the way the app is actually used.
  Launched with no console via `Studio Assistant.cmd` and the Desktop / Start Menu
  shortcuts.
- **`studio_tasks.py`** — shared GUI/CLI execution, original-schema validation,
  bounded request context, cancellation, execution journals and task recovery.
- **`studio_toolsmith.py`** — tools the model makes for itself, and the per-app
  library they are kept in.
- **`studio_mcp.py`** — the MCP harness. `Server` is the protocol every bridge written
  here runs on (framing, revision negotiation, validation, annotations, logging,
  progress, cancellation); `Loopback` is `MCPClient`'s interface over a `Server` in
  this process; `check_tools()` / `check_live()` hold any bridge — ours or installed —
  to what the executor and the inference host need. `python studio_mcp.py check --app
  <id>` is the command; see *The MCP harness* below.
- **`studio_comfy_mcp.py`** — our own MCP stdio bridge to ComfyUI's HTTP API. The
  one bridge written here rather than installed, because ComfyUI has no MCP server
  of its own and the stdlib-only rule bars the ones on PyPI. `--list-tools` prints
  its contract.
- **`studio_opencode_mcp.py`** — our MCP stdio bridge to an OpenCode server, which
  `ContainerSpec` runs in a Docker container built from `opencode/Dockerfile`. Its file
  tools are confined to the workspace folder the container is given. `--list-tools`
  prints its contract.
- **`studio_com.py`** — the road into an Adobe app that registers COM automation: a
  PowerShell worker holding `Photoshop.Application` / `Illustrator.Application`, an
  ExtendScript prelude (JSON serializer, error folding, unit pinning), and `ComHost.run()`
  which turns a script body into a decoded value or a `ComError`. See *The COM bridges*.
- **`studio_photoshop_mcp.py`**, **`studio_illustrator_mcp.py`** — our bridges to the
  Photoshop and Illustrator on this machine, each a table of tools whose bodies are
  ExtendScript run through `studio_com`. Nothing is installed inside either app.
- **`studio_cep.py`** — the road into an Adobe app that registers no COM: a CEP panel
  inside the app running a loopback HTTP server. `CepHost.run()` wraps a script body
  exactly as `studio_com` does and posts it; `install_panel()` copies a panel folder
  under `%APPDATA%\Adobe\CEP\extensions`; `explain_unreachable()` says the one
  thing to do when nothing answers. See *The CEP bridge*.
- **`studio_premiere_mcp.py`**, **`premiere_panel/`** — our bridge to Premiere Pro (the
  Beta this studio cuts in): a table of tools whose bodies are ExtendScript run through
  `studio_cep`, and the panel that evaluates them. `--install-panel` installs the panel.
- **`studio_research_mcp.py`** — the Chat tab's bridge: this PC's files and the web,
  read-only (`list_folder`, `find_files`, `read_file`, `search_web`, `fetch_page`). The
  one bridge the GUI runs *in process*, through `studio_mcp.Loopback`. See *The tab
  with no app*.
- **`studio_icons.py`** — reads an app's own icon out of its `.exe` (PE resource
  directory → `RT_GROUP_ICON` → `RT_ICON` → DIB or PNG → resample → PNG), and
  writes the PNGs `make_icon.py` packs into the `.ico`. `struct` and `zlib` only.

### Where the work happens

Inference is **remote**; tools are **local**. That split is not negotiable:

```
this PC (the workstation)                       tailnet peer
┌────────────────────────────────────┐         ┌──────────────────────┐
│ studio_chat.py / studio_agent.py   │  HTTP   │ LM Studio            │
│   ├ MCP stdio ─┐                   │ ──────► │ 100.127.17.38:1234   │
│   │            ▼                   │         │ OpenAI-compatible    │
│   │  @engine-room/after-effects-mcp│         └──────────────────────┘
│   │            │ ws 127.0.0.1:7777
│   │            ▼
│   │  CEP panel inside After Effects
│   │
│   ├ MCP stdio ─┐                   │
│   │            ▼                   │
│   │  davinci-resolve-mcp ── in-process ──►  DaVinci Resolve
│   │                                │
│   ├ MCP stdio ─┐                   │
│   │            ▼                   │
│   │  studio_photoshop_mcp.py ── powershell.exe ── COM ──► Photoshop
│   │  studio_illustrator_mcp.py ─ powershell.exe ── COM ──► Illustrator
│   │                                │
│   ├ MCP stdio ─┐                   │
│   │            ▼                   │
│   │  studio_premiere_mcp.py        │
│   │            │ http 127.0.0.1:7787
│   │            ▼                   │
│   │  premiere_panel (CEP) inside Premiere Pro
│   │                                │
│   ├ MCP stdio ─┐                   │         ┌──────────────────────┐
│   │            ▼                   │  HTTP   │ ComfyUI              │
│   │  studio_comfy_mcp.py           │ ──────► │ 100.127.17.38:8188   │
│   │                                │         └──────────────────────┘
│   └ MCP stdio ─┐                   │
│                ▼                   │
│      studio_opencode_mcp.py        │
│                │ HTTP 127.0.0.1:4096
│                ▼                   │
│   ┌ Docker container ────────────┐ │
│   │ opencode serve               │ │ ──────► LM Studio (same host as above)
│   │ /workspace ◄─ one folder     │ │
│   └──────────────────────────────┘ │
└────────────────────────────────────┘
```

Every bridge runs on this machine — two talk to a CEP panel (After Effects on 7777,
Premiere on 7787), one links against Resolve's scripting API in-process, two run
ExtendScript inside Photoshop and Illustrator over COM, and the ComfyUI one is a plain
HTTP client.
Only inference and image generation are remote, because the RTX 5090 here is reserved
for AE/Resolve rendering and must not be occupied by a resident model. ComfyUI is
therefore the one *app* that is not on this machine: the registry calls that
`remote`, below. OpenCode is on this machine but *not on its disk*: it runs in a Docker
container that is handed one folder, the registry calls that `container`, below.

## The app registry

`eng.APPS` is the single description of everything drivable. An `AppSpec` carries how
to find the app (`exe_globs`), how to tell it is running (`probe`), how to start its
bridge (`command`, `args`), what to expose (`groups`, `default_groups`), and how to
talk about it (`system_prompt`, `examples`, badge colours). Adding an app is an entry,
not a code change; `tests/test_agent.py` walks the registry and fails on a half-filled
one.

### The tab with no app: `ChatSpec`

`CHAT` is an `AppSpec` subclass with the *application* emptied out — no `exe_globs`,
no `probe`, `drivable = False` — but not the bridge. Its tools are
`studio_research_mcp.py`'s: list and search folders on this PC, read a text document
(plain text, code, JSON, a `.docx`'s paragraphs), search the web, read a page as text.
Every one is a read. It duck-types the rest of `AppSpec`, so `Session`, the tab strip,
the transcript and the executor need no special case for it. The rules that keep it
honest:

- **It is not in `APPS`.** `DRIVABLE` is derived from that list, and the sidebar must
  not advertise chat as something the agent can drive. `TABS` / `TABS_BY_ID` are the
  wider set — everything that can be a tab — and are what the tab strip, the new-tab
  menu and `--app` read.
- **`drivable` and `bridged` are two different questions.** `drivable` is "is there an
  application to find, probe, launch and repair" — `Start <app>`, `_fix()`,
  `_refresh_bridge()`'s not-running state, the sidebar rows all ask it, and chat says
  no. `bridged` is "is there an MCP bridge with tools" — starting one at boot, the
  library of made tools, the bridges row and its menu, the capabilities window, the
  CLI's bridge path all ask *that*, and chat says yes. Adding a `drivable` check to
  something bridge-shaped silently strips the chat tab of its tools; a test boots the
  chat tab and asserts it has them.
- **Its bridge runs in this process.** `AppSpec.connect()` is where a tab's bridge
  comes from — an `MCPClient` subprocess for every app — and `ChatSpec.connect()`
  returns a `studio_mcp.Loopback` over `studio_research_mcp.SERVER` instead: nothing to
  spawn, nothing to fail, no pipe to lose. `command`/`args` still name the script, so
  `python studio_mcp.py check --app chat --in-process` and `--call` work on it like any
  bridge written here, and `tests/test_mcp.py` walks it with the others.
- **The bridge is read-only by construction, and says no to two things.** Files that
  exist to hold secrets (`.ssh`, `.aws`, `.gnupg`, `*.pem`, `*.key`, `*.kdbx`, …) are
  refused by name, because a fetched page is untrusted text and "read this file, then
  fetch this URL" is the shape of an exfiltration; the prompt tells the model that what
  a page or file says is information, never instructions. And every walk and fetch is
  bounded — entries, seconds, bytes — and a result that stopped early *says so*, so a
  glob over `C:\` is a partial list and a sentence, not a hung tab. Long texts come
  back in windows whose first line names the next `start`; the engine clips a tool
  result at `MAX_TOOL_RESULT_CHARS` anyway, so the window default sits under it.
- **The search endpoint wants a browser.** `search_web` scrapes DuckDuckGo's HTML
  endpoint (`STUDIO_SEARCH_URL`) with a browser user agent; with the bridge's own
  name it answers a bot check and no results, and it is what a share of ordinary sites
  answer with 403. A bot page is recognised and reported as a refusal, not parsed into
  zero results. If the markup changes, `SearchResults` is the one class to fix and
  `tests/test_research.py` holds the shape.

Its prompt has one job the app prompts do not: **stop the model claiming work it
cannot do.** There is no app bridge to fail, so nothing else will contradict a
confident "done — I added the layer"; it also names every tool the tab has, so the
model looks rather than guesses. `chat_prompt()` drops `CHAT_SUFFIX` and swaps
`QUALITY_RULES` for `CHAT_RULES`: the app rules are about inspecting and verifying
edits, and this tab makes none. The read-only annotations are what keep the executor
from asking for a read-back after a `read_file`.

Two things stay derived, never hand-maintained:

- `DRIVABLE` is built from `APPS`, so the sidebar cannot advertise more than the agent
  can actually do. **Only add an entry when a bridge really exists.**
- `probe` is a strategy *string* (`"port:7777"`, `"process:Resolve.exe"`,
  `"url:http://host:port/path"`) rather than a callable, so the registry stays data a
  test can walk.

### The app on another machine: `remote`

An entry with **no `exe_globs`** is remote. ComfyUI is the only one: it lives on the
LLM PC, so there is no `.exe` here to find, no icon to read (the badge is drawn), and
nothing to launch. `installed()` is True for it — the tab is always worth offering —
`running()` probes its `url:`, and `launch()` raises with the `launch_note`, which for a
remote app has to say *where* to start it. The GUI asks `app.remote` before it offers
to start anything: the header button becomes **Check ComfyUI**, `_fix()` re-probes
instead of launching, and the status reads "not reachable" rather than "not running".
`tests/test_agent.py` holds a remote entry to all of that.

In the sidebar a remote app is a row like any other, but in its own group:
`detect_apps()` appends every remote registry entry with `"remote": True` and no
`exe`, and `_build_apps()` draws those last under a second heading, **ON LLM PC**
(`LLM_PC` in `studio_chat.py`), after the rows for this machine. Pinning orders a row
within its group rather than across them - a pinned ComfyUI is still on the other
PC, and the heading has to stay true. The heading appears and disappears with its
rows, so hiding ComfyUI hides the group.

The URL is `COMFYUI_URL` (default `http://100.127.17.38:8188`), read once in the
engine for the probe and the bridge label, and again by `studio_comfy_mcp.py` in its
own process — keep both reading the same variable. ComfyUI must be started with
`--listen` on that machine or it binds to its own loopback and the probe fails.

### The app the user connects by hand: `BridgeSpec`

Not every app has a bridge written here, and the user may already have one installed —
a Premiere Pro CEP bridge, a Blender server. `BridgeSpec` is an `AppSpec` built from a
command line the user typed, in the dialog `Chat._bridge_dialog()` builds (right-click
a non-drivable sidebar row, or *File → Connect an MCP bridge…*) or from `--mcp` on the
CLI. Two things about it are unlike every other entry:

- **It is filled in twice.** At construction it knows only what the user typed: name,
  command line, optional exe and probe. Its `groups` are `{}` and `tool_names()` is
  empty. When its bridge answers, `learn(tools, instructions)` derives groups from the
  tool names (`group_by_prefix()`: one level of prefix when that makes at least two
  families, else `all`) and keeps the bridge's `instructions`; `system_prompt` is a
  property that folds those instructions into `BRIDGE_PROMPT`. `_boot_bridge()` calls
  `learn()` and then **replaces `s.messages[0]`**, because `Session()` built the prompt
  from the empty entry and the warm-up must pay for the prefix a real message uses.
  The CLI does the same before choosing groups, which is why `--list-groups` cannot
  show a custom bridge's groups.
- **It joins the registry at runtime.** `add_bridge()` / `remove_bridge()` maintain
  `APPS`, `APPS_BY_ID`, `TABS` (chat stays last), `TABS_BY_ID` and `DRIVABLE` together;
  never append to one of those by hand. `DRIVABLE` is re-derived on every change, and a
  bridge written here keeps its sidebar row: a hand-entered bridge named "After Effects"
  gets a tab but not the row, and `_save_bridge()` refuses the name outright. Its id is
  the name's slug, suffixed `-bridge` if that would shadow a built-in id.

Entries live in the settings file under `bridges` as `record()` dicts, re-validated by
`bridge_from_record()` on load (junk is skipped, never fatal), and `custom` is the flag
the GUI asks before offering *Edit the bridge…* / *Forget the bridge*. `installed()` is
True (the user said so), `running()` is True with no probe (the bridge answering is the
evidence), and `launch()` needs an exe or explains that it has none. `tests/test_agent.py`
(`TestHandEnteredBridges`, and the GUI tests around `_save_bridge`) hold all of this.

### The app in a box: `container`

OpenCode is a coding agent — it edits and runs whatever it is pointed at — and this is
a production workstation. So `ContainerSpec` never runs it on the bare filesystem:
`launch()` builds `opencode/Dockerfile` into the image `OPENCODE_IMAGE` (once; the
first Start takes minutes), writes an `opencode.json` into the workspace that points
OpenCode at the studio's LM Studio, and runs the container with `docker_run_args()`.
Read that function before changing anything about isolation; it is the whole of it:

- **One bind mount.** `OPENCODE_WORKSPACE` (default
  `%LOCALAPPDATA%\StudioAssistant\opencode-workspace`) at `/workspace`, and nothing
  else from this PC. A named volume holds OpenCode's own session store so its history
  survives a restart. `tests/test_agent.py` asserts the mount list is exactly that.
- **Loopback only.** The port is published as `127.0.0.1:4096`, so nothing on the LAN
  reaches the server. `--cap-drop ALL`, `no-new-privileges`, a memory cap and a pid cap.
- **The bridge is confined the same way.** `opencode_put_file` / `read_file` /
  `list_files` resolve every path under the workspace with `realpath` and refuse one
  that lands outside it — `..`, an absolute path, a drive letter. A `/workspace/...`
  path is accepted and mapped, because that is how OpenCode names files in its
  replies. `tests/test_opencode.py` walks the escapes.
- **Nothing here reaches the creative apps.** The tab has no AE or Resolve tools, and
  the container cannot see their projects. The prompt says so, so the model does not
  offer.

In the registry it is a `ContainerSpec`: `exe_globs=[]` like a remote app, but
`remote` is overridden to False and `container` is True, so the GUI's `Start <app>`
button applies (`launch()` is what it calls), the sidebar lists it under **this PC**
with `container` where a year would go, and `installed()` asks whether Docker is here —
`docker_exe()` also looks in Docker Desktop's own folder, since a window launched from a
shortcut does not always have it on PATH. Neither Docker nor OpenCode is needed to run
the tests: `test_launch_builds_once_then_runs_the_container` swaps `eng.docker` for a
recorder.

`OPENCODE_URL` is read once in the engine (probe, bridge label, published port) and
again in `studio_opencode_mcp.py`; keep both reading the same variable, and the same
for `OPENCODE_WORKSPACE`. A loopback LM Studio host is rewritten to
`host.docker.internal` in the config, because `127.0.0.1` inside the container is the
container.

The server's routes live in one table, `ROUTES`, at the top of the bridge.
`opencode_status` fetches the server's own OpenAPI document (`/doc`) and names any
route in that table the document does not list, so an OpenCode release that renames
one is a sentence in the transcript rather than a 404 the model improvises around.
Older servers without `/global/health` are read from `/doc` instead.

The window does not stop the container when it closes, just as it does not close After
Effects. `docker stop studio-opencode` does; the workspace and the session volume stay.

### The COM bridges: Photoshop and Illustrator

Both apps register out-of-process COM servers on Windows whose `DoJavaScript` runs
ExtendScript inside the live app and returns the last expression as a string. That is
the entire bridge; no CEP panel, no UXP plugin, nothing to keep in step with an app
update. Python's stdlib has no COM client, so `studio_com.ComHost` keeps one
`powershell.exe -Sta` worker per app holding the COM object, and sends it one request
per line: the script file to run and the file to write the answer to. Things to know
before changing either bridge:

- **A tool is a script body.** `HOST.run(body)` wraps it in `studio_com.PRELUDE` (a JSON
  serializer — ExtendScript is ES3 and has none — plus `__px`, `__round`, `__fail`), each
  bridge's `HELPERS` (`__doc`, `__layer` / `__item`, `__info`, `__color`) and a
  `SETUP`/`TEARDOWN` pair, and decodes the JSON that comes back. `return` a plain value.
  A thrown error, from the script or from COM, arrives as `ComError` and the harness
  turns it into an `isError` result. Arguments go in with `json.dumps` (`J()`), never by
  string concatenation.
- **Units are pinned per call and restored.** Photoshop's `SETUP` sets ruler and type
  units to pixels and `displayDialogs` to `NO`; Illustrator's sets the coordinate system
  to the active artboard's and suppresses alerts. Without the first, `doc.width` comes
  back in whatever the user's rulers show. `UnitValue`s are recognised with `instanceof`
  — `typename` is not set on them.
- **Illustrator's y is flipped.** Illustrator counts y upward even in artboard
  coordinates; every helper negates it so the model sees y down from the artboard's
  top-left, the same as Photoshop and After Effects. `ai_run_jsx` is the one place the
  native convention leaks, and its description and the prompt say so. Adding an artboard
  makes it active in Illustrator; `ai_add_artboard` puts the previous one back unless
  asked, because "active" is what positions are measured from.
- **Never let a status tool touch COM.** `New-Object -ComObject` starts a closed app.
  `ps_status` / `ai_status` check the process with `tasklist` first and answer without
  attaching when it is down; every other tool is allowed to start the app, and `run()`
  extends its timeout by `LAUNCH_GRACE` when it does. A worker that stays silent past
  the timeout — a modal dialog inside the app — is killed and restarted on the next
  call, and the error names the dialog.
- **An unsaved document has a bogus `fullName`.** Both apps answer `fullName` for a
  document that was never saved (Illustrator points into `system32`); `__path()` checks
  the file exists before reporting a path, and `ps_save` / `ai_save` refuse and point at
  `save_as`.
- **Layers by `layer_id`, items by `uuid`, layers and artboards by name.** Photoshop's
  `Layer.id` and Illustrator's `PageItem.uuid` are stable across a session and what
  every edit tool takes; `getPageItemFromUuid` is used when present and a walk of
  `pageItems` when not. Names repeat; indices shift.
- **Screenshots are files returned as image blocks.** `ps_screenshot` duplicates,
  flattens, converts to 8-bit RGB, resizes and saves a PNG, then closes the duplicate
  and restores the active document; `ai_screenshot` exports the artboard with
  `ExportOptionsPNG24`. Both are `READ_ONLY`, so the executor treats them as the
  observation rather than nagging for one.

`tests/test_agent.py` (`TestComBridges`) covers the bridges with `HOST.run` replaced;
`tests/test_mcp.py` runs both (and the Premiere bridge) through `Loopback` against their
registry entries. One
test starts a real PowerShell worker with a ProgID nobody registered, to prove the COM
error path is a sentence and not a hang; nothing in the suite attaches to an app. The
live smoke — every tool against the real apps — is a script run by hand.

### The CEP bridge: Premiere Pro

Premiere registers no COM automation (`Adobe.Premiere.Pro.Beta.Project.27` and friends
are file types), so its road in is the one After Effects' bridge uses: a CEP panel that
runs a loopback HTTP server inside the app. Ours is `premiere_panel/` — a manifest, a
page and one `main.js` — and it is deliberately dumb: `POST /run {"script"}` evaluates
the ExtendScript through `window.__adobe_cep__.evalScript` and answers with the string
it returned. Every tool body, helper and the JSON serializer stay in Python, so the panel
never changes when a tool does, and `tests/test_premiere.py` can read the bodies as
text. Things to know before changing it:

- **The same wrapper as the COM bridges.** `CepHost.run()` calls `studio_com.script()`,
  so a body `return`s a plain value, `__fail()` is a sentence, and a thrown error comes
  back as `CepError`. `UnitValue` is core ExtendScript, so the prelude runs unchanged.
  Premiere's `Time` objects never reach the serializer: `__sec()` reads them as rounded
  seconds and `__time()` makes one; `__tc()` renders a timecode for the QE calls that
  want one (`razor`, `exportFramePNG`).
- **A call cannot start the app.** COM starts a closed Photoshop; a panel exists only
  while Premiere runs with it open. So `ppro_status` checks the process with `tasklist`,
  then pings the panel, and `explain_unreachable()` answers with the one thing missing —
  the process, the installed panel, CEP's `PlayerDebugMode` (checked through `winreg`,
  never set), or the panel not yet opened from *Window > Extensions*. The prompt tells
  the model to relay that text word for word.
- **Loopback only, JSON only.** The panel binds `127.0.0.1` and refuses a POST without
  `Content-Type: application/json`, which a browser page cannot send cross-origin
  without a preflight nobody answers — so a web page open on this PC cannot drive
  Premiere through it. The port is `STUDIO_PREMIERE_PORT` (default 7787), read by the
  engine for the probe, by the bridge for its URL and by `main.js` from Premiere's
  environment; keep all three reading the same variable.
- **The panel is unsigned.** CEP loads it only with `PlayerDebugMode=1` under
  `HKCU\Software\Adobe\CSXS.11` and `.12`; this workstation has both. `--install-panel`
  prints the `reg add` lines for any that are missing rather than running them. Premiere
  reads the extensions folder at startup, so install with it closed. A double hyphen
  inside an XML comment is illegal, and CEP refuses the whole manifest over one; a test
  parses it.
- **Ids are `nodeId`s.** Timeline clips go by `clip_id`, project items by `item_id`, both
  Premiere's own `nodeId`; sequences by name. `ppro_razor` changes the ids on the tracks
  it cuts and its result says so; `ppro_add_to_sequence` diffs the id set before and
  after to report what landed, because Premiere's insert methods return a boolean.
- **QE is used where the main DOM has no verb** — cutting, adding an effect, exporting a
  frame. `__qeclip()` matches the QE item by name and start ticks rather than by index,
  because QE's `getItemAt` counts gaps. A wrong match is a sentence pointing at
  `ppro_run_jsx`, not a cut in the wrong place.
- **Export presets are files.** `ppro_export` takes a `.epr` path or a preset name and
  resolves the name through `list_presets()`, which globs Premiere's and Media Encoder's
  `systempresets` folders and the user's; the folder name's last eight hex digits are
  the format's four-character code (`48323634` is `H264`). An ambiguous name lists the
  matches and refuses. Rendering in Premiere blocks it and the call waits (`EXPORT_TIMEOUT`);
  `queue=true` hands the job to Media Encoder instead.
- **Learned against the real Premiere 27 Beta, and held by tests:**
  `app.project.createNewSequence()` opens the *New Sequence* dialog and every later
  `evalScript` queues behind it — the panel's GET still answers, so it looks alive while
  nothing runs. Sequences are therefore made from a `.sqpreset` through QE's
  `newSequence()`, which is silent, and `setSettings()` then applies width/height/fps
  (any size works; HD 1080p at the nearest fps is the base). QE takes paths only as
  `File(...).fsName` — a forward-slash string returns false or "Unknown error" — so every
  path handed to Premiere goes through `fs()`; `exportFramePNG` appends `.png` itself.
  `marker.end` takes a number of seconds and refuses a `Time`. `getSpeed()` is a ratio.
  Premiere 27 renamed the old blur to "Gaussian Blur (Legacy)"; the new "Gaussian Blur"
  has different parameters, and effect parameter lists carry blank and `_ `-prefixed
  internal names that `__shown()` drops. An unset sequence in/out reads as a negative
  sentinel, mapped to null. And `tasklist`'s table view truncates image names at 25
  characters — `Adobe Premiere Pro (Beta).exe` lost its `.exe` and the running check
  failed — so both `process_running()` helpers ask for CSV.
- **`tests/test_premiere.py` starts a fake panel** on a random loopback port to prove the
  transport — the wrapper, the decode, a silent panel becoming a "dialog may be open"
  sentence, nothing listening becoming "not running" — and a temp `%APPDATA%` to prove
  the install; nothing in it touches Premiere. The live smoke — every tool against the
  real app — is done by hand with `python studio_mcp.py check --app premiere --call`.

### What a `system_prompt` has to carry

The model driving these apps is small and local. It knows After Effects and Resolve in
general; it does not know *this bridge* at all. The prompt is where the bridge's own
conventions live, so each app's covers, in this order:

- **How the project is shaped** — the object graph, and what addresses each object.
- **Units — the ones that fail silently.** AE is seconds, RGB 0..1, opacity 0..100,
  scale in percent, origin top-left. Resolve is frames and timecode, `track_index` from
  1, `item_index` from 0. Premiere is seconds, tracks from 1, and Motion's Position is
  normalised 0..1 across the frame. A model left to guess these produces something that
  renders happily and is wrong.
- **How to make and change things** — the default each creating tool applies, and when
  to override it.
- **What to do when a tool cannot reach the app** — one path, and no retry loop.

Two rules for editing them:

**Check every fact against the bridge's tool schema, not against knowledge of the app.**
The Resolve prompt used to say to start from `media_pool` action `list`. `media_pool`
has no `list` action — clips come from `folder` `get_clips`. Nothing catches that at
runtime except a failed call and a model that improvises around it.

**Only name tools the app actually exposes.** `default_groups` is the working set;
naming a tool outside it strands the instruction and the model answers by inventing a
call. `tests/test_agent.py` walks each prompt for tool names and fails on one that is
not exposed. ComfyUI's `workflows` group (arbitrary API-format graphs and the node
catalogue) is off by default for exactly this reason: a 3B-active model handed the
whole node catalogue builds broken graphs, and `comfy_generate` covers the ordinary
work. `tests/test_comfy.py` also asserts the bridge's tool list and the registry's
groups are the same set, so a tool added to one and not the other fails loudly.

Length is not a per-message cost. The prompt is the head of every request's prefix, so
LM Studio caches it after the first call and the warm-up pays for it against the exact
prefix a real message uses — once per tab, not once per question.

## Hard-won constraints — read before editing

**Stdlib only.** No `openai`, no `mcp`, no `requests`, no pip step. This runs on a
workstation where a broken Python environment costs real production time. Tkinter is
the GUI for the same reason. Do not add dependencies.

**`sanitize_schema()` is load-bearing.** The AE bridge describes fixed-length arrays
(`bgColor`, `position`) in JSON Schema 2020-12 tuple form: `prefixItems` next to
`"items": false`. LM Studio's schema-to-grammar converter is draft-07 shaped and
rejects the boolean with `Unrecognized schema: false` — **HTTP 400 for the entire
request**, not just that tool. It hit 13 of 45 tools. Any other OpenAI-compatible
runner (llama.cpp, vLLM grammar mode) will hit the same wall.

**Never clip a tool description short.** Resolve's 27 tools are *compound* — every one
is `(action, params)`, and the description is the entire list of actions it accepts.
The old 1024-char cap amputated half of `timeline`'s actions, and the model then
confidently invented the missing ones. That failure is silent: no error, just wrong
calls. `MAX_TOOL_DESC_CHARS` remains 4000 as a compatibility constant, but conversion
now preserves full descriptions. The executor budgets the entire request and fails
explicitly if the fixed tool contract and brief cannot fit.

**Never name a Tk widget method `_w`.** Tkinter's `Misc` uses `self._w` for the
widget's Tcl pathname. Shadowing it sends `__repr__` into infinite recursion and the
window never opens. `Misc._bind` and `Misc.quit` are the same kind of trap. Audit new
method names against `tkinter.Misc` / `tkinter.Tk` — a test does it for you.

**The tab strip folds; it never overflows.** Seven labelled tabs are wider than the
strip at the window's minimum size, and Tk's packer answers by pushing the last ones
off the edge, unmapped and unreachable. `_fit_tabs()` measures the labelled row from
its parts' requested widths — not from the packed tab, which needs an idle pass, and an
idle pass from inside the `<Configure>` handler re-enters the method (it did; the test
suite went from 4 s to 56 s) — and when it does not fit, every tab but the active one
drops to mark and dot, the mark carrying the name as a tooltip. Called on add, close,
select and resize. A test asserts every tab stays mapped at the minimum width.

**Tk pack order: fixed-size widgets first.** An expanding sibling packed *before* a
fixed one claims the leftover space and pushes it off the edge. This bug has shipped
three times — hiding the Send button, hiding the whole composer until the window was
resized, and then squeezing the pin, hide and status dot off every sidebar row. In
`_build()` the composer is packed first, then the tab strip, then the expanding
transcript stack; inside the composer the Send button precedes the input; in
`_app_row()` the mark, both glyph buttons and the dot all precede the name box.
`tests/test_agent.py` asserts the invariant; keep it passing.

**Colours are palette roles, not hex.** Preferences repaints the *running* window —
a rebuild would throw away every transcript — so each widget is registered in
`Chat.skin` with the role each of its colours came from (`bg="side"`, `fg="faint"`),
and `_theme()` re-reads them. Two consequences worth remembering: anything put on the
queue from a worker thread carries a role name rather than a colour (`("status", sid,
("connected", "ok", False))`), and a canvas that draws itself — dot, app mark — has
to be repainted rather than reconfigured, which is what `dot_role` and `marks` are
for. A role missing from either palette is a `KeyError` mid-switch; a test compares
the two.

**Fonts scale with the display, pixel counts do not.** Tk sizes fonts from the
screen's DPI, so on this 150% workstation every label is half again as wide while
`width=236` stays 236. That combination clipped the sidebar. Anything measured in
pixels goes through `Chat._px()`, and `_metrics()` sizes the rail against the widest
row it is actually going to draw. Design at 96dpi, multiply on the way out.

**Do not remove the startup warm-up.** It looks like a redundant throwaway request.
It is not: a full tool-schema set takes about a minute to prefill cold. The warm-up
pays that against *the exact prompt prefix a real message uses*, so the first question
returns in seconds. LM Studio's prefix cache survives across processes, which is why
this works at all. Each tab has its own prefix and so warms up separately, the first
time it is opened.

**Never test the GUI with synthetic keystrokes.** `SendKeys` types into whatever
window has focus, not the one you meant. It has already leaked a test sentence into
the user's chat window mid-run. Test by constructing `Chat()` in-process and calling
its handlers, and assert on widget geometry for layout. Screen-capturing the window
to look at it is fine — that's read-only.

## Tabs and sessions

One `Session` per app: its own `MCPClient`, tool list, message history, transcript
`tk.Text` and busy flag. Nothing is shared but the `LLM` (one host, one model) and the
composer.

- **Worker events carry a session generation**: use `s.event_id` (app id, unique
  generation) in `self.q.put((kind, sid, payload))`. App IDs still key UI dictionaries.
  Never route worker output with only the reusable app ID. A `sid` of
  `None` means "whatever tab the user is looking at" — startup failures on the shared
  inference host have no app of their own. `_handle` resolves it.
- **The header describes the active tab only.** Status, the Send button and the
  `Start <app>` button all come from `_apply_status()` reading the active session, so a
  background tab finishing work never rewrites the header you are looking at.
- **Boot is lazy and idempotent.** `_ensure()` fires on first `_select()`; `booting`
  guards re-entry and is cleared in a `finally` so a failed bridge can be retried by
  switching away and back.
- **Tabs come and go.** `_add_tab()` / `_close_tab()` maintain `sessions`, `order`
  and `tab_ui` together, and closing shuts the MCP subprocess down off the UI thread.
  Closing sets cancellation. An operation already in flight may finish;
  `_handle()` drops events for missing or mismatched generations, including when
  the same app tab has already been reopened.
- **Every tab can be closed.** `active` is then `None` and `cur()` returns `None`;
  the stack shows `self.empty`, and host-level errors — which have no app of their
  own — are written into `empty_msg` so the rule that errors reach the user as prose
  still holds with nothing open.
- **Which tabs are open is a preference**, not a fact about the machine. Settings
  live in a small JSON file under `%APPDATA%` (`STUDIO_SETTINGS` overrides the path,
  which is how the tests avoid the user's real one). Reading and writing it are both
  best-effort: a hand-wrecked or unwritable file costs a preference, never the app,
  and everything loaded off disk is re-validated — `"hidden": "nope"` must not hide
  four apps called n, o, p and e.

## Task execution and recovery

- GUI and CLI use `studio_tasks.Executor`; do not add another tool loop. That
  includes the chat tab: same executor, same journal, an empty tool list.
- Keep original MCP schemas in `Session.schemas` for validation. Sanitized schemas
  are the inference representation only; preserve compound-tool descriptions.
- `studio_task_update` is an internal tool exposed alongside bridge tools. Warm-up
  must use the same system prompt and tool list, including that internal tool.
- GUI task JSON files live beside settings in `tasks/<app>/<task-id>.json`. Persist
  intent before dispatch and results afterward. A disk failure stops edits. Task
  records preserve conversational progress, not project backups or undo state.
- Unknown write outcomes must not trigger blind repeats. Restored tasks need a
  project read before editing. Read-back guards do not independently prove visual
  or semantic correctness; distinguish observations from model claims.
- `readonly()` decides which calls owe a read-back from the tool's name prefix or its
  MCP `readOnlyHint` annotation. A bridge written here must annotate its reads
  (`READ_ONLY` in every bridge here); unannotated, `comfy_status` counted as an
  edit and the model was nagged to "inspect" until it cycled on `studio_task_update`.
  A successful call that returns an image has done its own read-back — a generate
  that waited for its files is the observation, not something to inspect afterwards.
- Stop prevents subsequent dispatches; it cannot undo or guarantee cancellation of
  an in-flight operation. Complete tool-result envelopes when stopping a batch.
- An `AppSpec` may carry `models`, small models it prefers, best first; ComfyUI does,
  because a 30B model resident beside a diffusion model on the same GPU is VRAM the
  pictures could have had. `AppSpec.model_for(ids, shared)` resolves it against what
  the host serves — `STUDIO_MODEL_<APP>` pin, then the list, then the shared model,
  never a model that is not served — and returns a note the tab prints once. The GUI
  keeps one `Session.llm` per tab, fixed at boot: the executor, both warm-ups and the
  host's cached prefix must agree, and tabs with no preference share the window's
  handle (`Chat._llm_for`). The CLI resolves the same way unless `--model` is given.
- **The executing model reads text; `eng.Vision` is its eyes, and every tab has
  one or is told it does not.** Every bridge answers a screenshot with an image
  content item and every tab takes picture attachments, so `_boot_host` resolves
  a vision model beside the executing one — `probe_models` now also returns the
  served ids that can take a picture (LM Studio types them `vlm`; a plainer host
  is judged by `looks_vision()` name hints), and `pick_vision_model` prefers
  `STUDIO_VISION_MODEL` when served, then the executing model itself if it can see
  (no second model in VRAM), then one already loaded (no load), then
  `PREFERRED_VISION_MODELS`, then anything the host has downloaded. Not-loaded is
  fine: `Vision.needs_load` says so and `load_model` posts to LM Studio's
  `/api/v1/models/load` — the GUI does it on the worker after `host_ready` so the
  tabs boot meanwhile, the CLI before the task; a host without that endpoint
  loads just-in-time on the first chat call, so a load failure is a line in the
  tab, never a stop. Pictures go through `studio_icons.flatten_png` first: a vision
  model sees alpha as black, so a black glyph on a transparent PNG - most logos,
  and an Illustrator artboard exported without a background - was being described
  as "entirely black"; it is composited onto white, opaque PNGs pass through
  untouched, and anything the decoder cannot read goes as is. The CLI resolves
  the same way. One `Vision` on the same remote host serves every tab
  — never move inference to the workstation — and `Executor(vision=Vision.review)`
  gets a review of every returned frame; `Vision.describe_all` is what `_turn`
  appends to a brief. When nothing served can see, `resolve_vision` returns the
  reason: the Inference row goes amber, `_boot_session` prints it once per tab,
  and `_turn` says it again whenever pictures are attached. Session-owned preview
  images keep Tk references alive; stale-generation preview events must be dropped.
- **What the user attaches never enters the messages as bytes.** The file and
  folder glyphs, Ctrl+O and Ctrl+Shift+O on the shared composer queue paths
  (`Chat.attachments`, one chip each — any file, any size, or a folder); `_on_send`
  appends `attachment_note()` to the brief — name, size, dimensions read from the
  header by `image_dims()` when it is a picture, and the path every bridge on this
  PC opens files by; a folder gets a listing capped at `LIST_LIMIT` entries so the
  model can name a file in it without a tool call — and shows a picture in the
  transcript where Tk can decode it. Base64 in a message would blow
  `context_messages`' character budget and land in every checkpoint, so what a
  picture *shows* comes from `Chat.vision`: `_turn` asks it for a description on
  the worker — pictures only, `is_picture()` decides — and appends that to the same
  brief before the executor starts, so it survives resume. With no vision model
  the model has the path only, and both it and the user are told so. `ATTACH_LIMIT` applies to
  pictures alone, because theirs are the only bytes anything reads. A
  `ContainerSpec` tab sees one folder: `attachment_note` copies the file (or the
  folder, whole) into `<workspace>/attachments/` and names the `/workspace/...` path
  the container will see.

## Tools the model makes for itself

`studio_tool_create` lets the model name a run of calls it keeps repeating.
`studio_toolsmith.py` holds the definition, the per-app library and the checks.

- **A made tool is data, never code.** It names tools already exposed to the tab and
  fills their arguments from its own declared inputs; `{input}` on its own keeps the
  input's type, anywhere else it is textual substitution. There is no `eval` here and
  there must not be one — this runs on the workstation driving a live project.
- **Steps go back through `Executor._call`.** That is the whole safety argument: a made
  tool cannot name a tool the tab was not given, cannot skip validation against the
  bridge's *original* schema or the compound-action contract, cannot get past the
  Resolve `quit` prohibition, and cannot keep its steps out of the journal — where each
  carries `via` naming the tool it ran under. Never dispatch a step any other way.
- **Made tools go last in the tool list.** `inference_tools()` appends them after the
  fixed contract so making one re-prefills the tail of LM Studio's cached prefix rather
  than the whole tool set. Keep them last, and keep every warm-up path passing the same
  library the executor gets, or the warm-up stops matching the request it is warming.
- **Creation checks the template with sample inputs**, so a structurally wrong step —
  a missing required argument, a misspelled key, the wrong type — is refused before the
  tool exists. `relax()` drops value constraints for that check on purpose: a sample
  cannot satisfy an enum or a pattern, and rejecting on one would block a legitimate
  tool. Real values are validated on every call like any other tool's.
- **The library is re-validated on load, against the tools the tab currently offers.**
  Groups change; a tool built on something no longer exposed is left out and said aloud,
  never handed to the model as a tool that cannot run. Like the settings file, a wrecked
  or unwritable file costs one made tool and never the app — and when it cannot be
  saved, the model is told the tool is session-only rather than left to assume.
- Every tab with a bridge gets a library, chat's included — `Chat._session()` is the
  one place a `Session` is made, at startup and from the new-tab menu alike. (Restored
  tabs once got none, and the tool maker worked only in tabs opened later.) A tab with
  no tools at all still gets none: `inference_tools([])` is `[]`.

## The MCP harness

`studio_mcp.py` is the one place the protocol lives. A bridge written here is a table
of `(name, fn, description, schema)` rows and a `Server`; the two bridges in this
folder are exactly that, and their `serve()` loops are gone.

- **`Server.handle()` is the protocol as a pure function.** A message in, a reply (or
  `None`) out. `serve()` only adds streams, a reader thread and a write lock. Test
  the protocol through `handle()` or `Loopback`; test stdio only for the things
  stdio adds — the parse error, the stdout guard, a cancel arriving mid-call.
- **Revisions are negotiated, not pinned.** `PROTOCOL_VERSIONS` is what the harness
  speaks, newest first; `initialize` echoes the client's revision when it is one of
  them and offers the newest otherwise. `MCPClient` asks for the newest and keeps
  whatever the bridge answers on `protocol_version`, `server_info`, `instructions`
  and `capabilities`. Both installed bridges answer 2025-11-25 today. Add a revision
  to the tuple when a feature here needs one; never pin.
- **Two kinds of "no".** Unknown tool, arguments the schema refuses, a parse error, a
  JSON-RPC batch: protocol errors, with the JSON-RPC code the spec names, because a
  caller that sends them skipped the executor's own validation. A tool's own refusal
  — `ComfyError`, `OpenCodeError`, the classes a bridge lists in `errors=` — is an
  `isError` result in the bridge's words, and any other exception is an `isError`
  result naming it with the traceback on stderr. The server keeps serving through
  all of it. The bridges' Python-level `call_tool()` folds the first kind into a
  result too, for callers that are not on the wire.
- **The validator is shared.** `studio_mcp.validate` is what the executor runs
  before a call and what a `Server` runs on arrival; `studio_tasks.validate` is the
  same function. One validator, so the two sides cannot disagree about a schema. It
  caught a test calling `comfy_generate` with a `timeout` under the schema's minimum
  the day it went in.
- **Annotations are the spec's defaults unless a tool says otherwise.** A `Tool` is
  assumed to write, to be destructive and to touch the outside world; `read_only=True`
  sets the three hints a read implies, and `HINTS` in each bridge overrides the rest
  (a generate is not destructive; a clear-queue is). `readOnlyHint` is the one the
  executor reads — see *Task execution and recovery* — so a read left unannotated
  is nagged for read-backs.
- **stdout is the wire.** `Server.call_tool` swaps `sys.stdout` for `sys.stderr`
  around the handler, so a `print` inside a tool reaches the client's log rather than
  the middle of a reply. Both stdio streams are reconfigured to UTF-8 with `\n`
  newlines before serving; the console default on this workstation is cp1252.
- **Progress and cancellation are opt-in per tool.** `studio_mcp.progress()` sends
  `notifications/progress` on the client's `progressToken` and does nothing without
  one; `studio_mcp.cancelled()` is True once the client sent `notifications/cancelled`
  for the call in flight. The reader thread acts on cancels while the main thread is
  inside a tool, which is the only reason it is a thread. `comfy_wait` polls both.
  `MCPClient` sends a cancel when it gives up waiting; a bridge built here then
  stops and never replies to a request nobody owns. A cancel for a request that
  already finished is ignored, as the spec asks.
- **`check` judges a tool list the way the executor and the host will**, and nothing
  else: the sanitizer's rewrite, `'items': false` surviving it, descriptions over the
  budget, keywords the grammar converter does not enforce, hints that contradict a
  name, groups naming tools the bridge lacks, tools no group exposes, prompts
  teaching a tool outside the working set, and the executor's own arithmetic —
  brief plus contract off `REQUEST_CHARS`, the rest for conversation. Pass the
  engine's `sanitize_schema` and `readonly` in; the module does not import the
  engine, so a bridge process can check itself with `--check`.
- **Installed bridges are held to recordings.** `python studio_mcp.py snapshot --app
  <id> tests/contracts/<id>.json` writes what the bridge exposes, and
  `tests/test_mcp.py` checks every recording against the registry's groups and
  prompts — offline, with the app closed. Re-record when a bridge updates; a
  recording older than the bridge is a test that passes for the wrong reason. The AE
  recording shows 14 tools no group exposes (markers, house style, jobs, the issue
  journal, `delete_comp`, `init_project`, `setup_panel`); some of that is deliberate
  and the rest is a decision nobody has made yet — the check will keep saying so.

## Running and testing

```bash
python -m unittest discover -s tests -v      # no network, no apps needed
python studio_agent.py --list-groups         # registry sanity, no bridge started
python studio_agent.py --app resolve --list-tools   # needs the Resolve venv
python studio_agent.py --app comfyui --list-tools   # no ComfyUI needed for the list
python studio_comfy_mcp.py --list-tools      # the bridge's own contract
python studio_comfy_mcp.py --check           # ...held to the harness's checks
python studio_opencode_mcp.py --list-tools   # likewise; no Docker needed for the list
python studio_photoshop_mcp.py --check       # the COM bridges; no app is touched by --check
python studio_illustrator_mcp.py --list-tools
python studio_premiere_mcp.py --check        # the CEP bridge; no app is touched by --check
python studio_premiere_mcp.py --install-panel   # copy premiere_panel/ under CEP/extensions (Premiere closed)
python studio_research_mcp.py --check        # the Chat tab's bridge; reads nothing by itself
python studio_mcp.py check --app chat --in-process --call   # ...and list_folder on the home folder
python studio_agent.py --app chat "find the brief in my Documents folder"   # the chat tab from the CLI
python studio_mcp.py check --app premiere --in-process
python studio_mcp.py check --app photoshop --in-process   # against the registry entry
python studio_agent.py --mcp "npx -y some-mcp" --name Blender --list-tools  # any bridge
python studio_mcp.py check --app comfyui     # start a registry bridge, report on it
python studio_mcp.py check --app after-effects --call   # ...and call its harmless reads
python studio_mcp.py check --app resolve --snapshot tests/contracts/resolve.json
python studio_mcp.py snapshot --app resolve tests/contracts/resolve.json  # re-record
python studio_chat.py                        # the real app (console attached)
```

`tests/` never touches the network, the creative apps, Docker, or the model — the
research bridge's `urlopen` is swapped for a fake with a table of pages,
the COM bridges are tested with `HOST.run` replaced, the one PowerShell worker a test
starts is given a ProgID nothing answers to, and the Premiere bridge talks to a fake
panel on a random loopback port —
`test_comfy.py` and `test_opencode.py` swap `urllib.request.urlopen` for an in-memory
server that answers the routes the bridge uses, and the OpenCode one works in a temp
workspace. `test_mcp.py` drives both bridges through `Loopback` — the real server
objects, in process — and holds the installed bridges to `tests/contracts/`. Tests that would
need a display skip themselves when there isn't one; the GUI tests stub
`installed_apps` so tab behaviour doesn't depend on what this machine has, stub
`_read_icons` so no .exe is opened, and point `STUDIO_SETTINGS` at a temp file so the
run cannot write to the settings the user's own window is reading. A menu is built by
one method and posted by another (`_menu_tabs()` / `_tab_menu()`) precisely so its
contents can be asserted — a posted menu grabs the pointer.

To exercise the live path the app must be **open** — AE's bridge binds 7777 only while
AE runs, and Resolve's server can only attach to a running Resolve. AE's `check_setup`
(via the MCP server) diagnoses the whole chain and its `nextSteps` are reliable; relay
them rather than guessing.

## Conventions

- Tool groups keep the exposed tool count down; a 3B-active MoE gets sloppy shown
  everything at once. Each app's `default_groups` is its working set.
- Identify AE layers by `id`, never `index` — an index shifts on every insert. In
  Resolve, media pool clips are `clip_id`; timeline clips are positional —
  `track_type` + `track_index` + `item_index`, the first counting from 1 and the last
  from 0. Photoshop layers are `layer_id`, Illustrator items are `uuid`, Premiere
  timeline clips are `clip_id` and project items `item_id`. The system prompts say all
  of this; keep them saying it. A test asserts the AE half.
- **The Resolve prompt forbids `resolve_control` action `quit`.** The tool exists and
  works; closing the user's Resolve mid-session costs unsaved work. A test asserts the
  prohibition is still in the prompt.
- App marks are the app's **own icon, read live out of its `.exe`** by
  `studio_icons.py`, with the drawn two-letter badge as the fallback whenever that
  fails — no resources, a packed exe, a path we cannot open. Nothing is shipped or
  cached on disk, so a new Adobe year needs no new art. Reading them is done on a
  worker thread (`_read_icons`): a cold read of a 500MB Photoshop binary is not
  something to do while the window is opening, and `PhotoImage` must be built on the
  UI thread anyway — and *kept*, or Tk drops the image when Python collects it.
  The shortcut icon is still drawn rather than shipped: `make_icon.py` generates
  `studio-assistant.ico` through the same PNG writer, so the mark changes by editing
  colours rather than by opening a paint program.
- Small controls (pin, hide, close, the bridges link) are single characters from
  **Segoe MDL2 Assets**, Windows' own icon font, by codepoint — same reasoning as the
  badges, and there is an ASCII fallback if the font is ever missing. Write them as
  hex codepoints: pasted into an editor the characters themselves are blanks.
- Errors must reach the user as prose. `_guard()` catches everything off the UI thread;
  tracebacks go to `studio_assistant_error.log`, never to the screen.
