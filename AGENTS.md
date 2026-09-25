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
  Launched with no console via `Studio Assist.cmd` and the Desktop / Start Menu
  shortcuts.
- **`studio_tasks.py`** — shared GUI/CLI execution, original-schema validation,
  bounded request context, cancellation, execution journals and task recovery.
- **`studio_toolsmith.py`** — tools the model makes for itself, and the per-app
  library they are kept in.
- **`studio_lessons.py`** — what the model learns per app: the `Notebook` of one-line
  lessons, the `studio_remember` tool, the reading of the user's corrections and the
  end-of-task reflection. See *What the model learns, asks and looks up*.
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
- **`studio_procs.py`** — every child process the app starts, contained: each in its
  own kill-on-close Windows job object, so it and everything it starts end with the tab,
  the app, or the app's crash. See *Processes: nothing outlives the app*.
- **`studio_milanote.py`** — the Milanote tab, which holds a window and has no bridge: a
  Chrome/Edge `--app` window re-parented into the tab, and uploads dropped onto the board
  over DevTools. See *The tab that holds a window*.
- **`studio_imagegen.py`**, **`studio_images_ui.py`**, **`comfy_workflows/`** — the
  Image Studio tab: a form (person, style, scene, references, generate) over any number
  of ComfyUI backends, with no model in the loop. See *The Image Studio*.
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
LLM PC, so there is no `.exe` here to find, no icon to read (its logo is drawn instead,
by `studio_icons.comfy_png`, keyed by app id in `DRAWN`), and
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

### What ComfyUI makes, and why it takes the time it does

The bridge has four making tools. `comfy_generate` is text to image, faces redrawn;
`comfy_edit_image` changes a picture by instruction (Qwen-Image-Edit 2509 with up to two reference
pictures); `comfy_face_swap` puts one picture's people's faces on another's;
`comfy_upscale` enlarges one and redraws its detail. Each takes a local path
and uploads it (`input_image`), so the model never spends a round trip on
`comfy_upload_image`. The recipes are ComfyUI's own templates, read from
`/templates/<name>.json` on the server: `image_z_image_turbo`,
`image_krea2_turbo_t2i_int8`, `image_qwen_image_edit_2509`,
`utility_z_image_turbo_2k_upscaler.app`. Check a new model there before guessing its
recipe; the wrong encoder type or latent gives noise, not an error.

- **`FAMILIES` order is the text-to-image preference, and edit models are never the
  default.** The LLM PC has Z-Image Turbo, Krea 2 Turbo and Qwen-Image-Edit as split
  models and only a SAM checkpoint. `plan_model` used to take the first known split
  model by filename, which was `qwen_image_edit_2509` matching the `qwen_image`
  family: every "draw a duck" ran a 20B edit model at 20 steps and CFG 2.5 (two
  passes a step) with no picture to edit. Slow, and soft. Now a known family beats a
  checkpoint, `FAMILIES` order ranks them (Z-Image, the photographic one, first), and
  a family marked `edit` is skipped.
- **Realism is three things, all on by default.** A photographic prompt (one naming
  no other medium, `NOT_PHOTO`) gets `PHOTO_SUFFIX`; a checkpoint at CFG above 1 with
  no negative gets `PHOTO_NEGATIVE`; and a family with a `hires` recipe gets the
  detail pass (`detail_pass`): lanczos ×1.5, VAE encode, resample at denoise 0.33 —
  the Z-Image upscaler template without ESRGAN, which is not installed. The pass is
  what turns smeared 1-megapixel micro-detail into pores, fibres and knots; measured
  on the LLM PC it added 20-55 s to a ~200 s job. It is capped near 4.2 MP and skipped
  for image-to-image. The prompt tells the model to describe a photograph in
  sentences and never write "masterpiece, 8k".
- **At CFG 1 the negative is a `ConditioningZeroOut` of the positive**, as the model's
  templates do; the sampler never reads it, so encoding a negative wasted time.
- **The GPU is shared, and that is where the time goes.** ComfyUI and LM Studio share
  one 24 GB RTX 3090. With the 30B executor and the vision model resident, ComfyUI saw
  0.3 GB free and staged every model through system RAM ("prepared for dynamic VRAM
  loading" in its log): a 1024² Z-Image picture took 200-330 s, of which 35 s was
  sampling, and the first Qwen-edit step took 140 s, and one render sat in "Model
  Initializing" until it was interrupted. The graph cannot fix that; making room can.
  An `AppSpec` lists its `gpu_tools`, the GUI (and the CLI) wraps that tab's
  bridge in `YieldGPU`, and around each of those calls `Chat._make_room` has
  `eng.make_room` unload every model on the host - the tab's own too, which only
  waits while the picture is made - and `Chat._give_back` reloads the tab's model
  at the context length it had (`eng.give_back`), even after a failed render. A
  model another tab is mid-request on stays.
  Keeping the tab's 9B resident was tried first: the 19.5 GB Qwen edit model then
  took 272 s for its first of four steps, and a Z-Image render sat six minutes in
  the VAE at the detail-pass size. Measured on the same 1248x1824 picture: 254 s
  with everything resident, 64-68 s with the vision model and the 30B gone, 41-49 s
  with the whole GPU and the text encoder on the CPU (below); an edit went from a
  272 s first step to 119 s in all. ComfyUI cannot see LM Studio's memory on this
  Windows host (its `vram_free` said 22 GB with the 30B loaded), so it never holds
  back: one 1024² render with only the vision model left in LM Studio sampled at
  18 s a step instead of 0.9 and took 165 s instead of 24. Making room is not
  optional.
- **ComfyUI lets go of the GPU when a run is done** (`release`, in `run()` and
  `comfy_wait`). It keeps its last run's models on the card until its own next
  run needs the room, and LM Studio cannot see them either: after one Z-Image
  render 11.7 GB were still held, reported nowhere, and every LM Studio model
  after it paid - the 30B decoded at 25-30 tokens a second instead of 73-81, the
  9B took 17 s to load instead of 3, the vision model ~30 s instead of 5. A run
  with nothing queued behind it now posts `/free` (`unload_models` and
  `free_memory`) and waits, `RELEASE_WAIT` at most, for ComfyUI's free VRAM to
  come back near where it was before the run - about a second - because the
  tab's model is loaded the moment the result is back. It costs the next render
  nothing: 24.3 s with the models kept, 23.6 s after `unload_models`, 23.4 s after
  both flags. `COMFYUI_KEEP_MODELS=1` keeps them, for a ComfyUI with a card of
  its own.
- **The picture comes back before the model does, and the vision model looks
  at it alone.** `YieldGPU` no longer gives back when the tool returns: it
  remembers what the render sent away, and `settle()` brings it back when the
  model is next needed - the executor calls `eng.settle` before every request
  to the model and when a run ends, and `Chat._turn` before it looks for the
  tab's model. Timed end to end, the finished picture used to sit unseen for
  17 s behind the 9B's reload. The order after a render is now: the picture,
  the vision model's check on an empty card (JIT, 7.5 s), then `_give_back`
  unloads the vision model and loads the tab's model alone (3 s). LM Studio
  loads a model onto an empty card in 3-5 s and beside another in 9-16 s -
  whichever comes second - and loading both at once made the check take 15 s
  instead of 8; this order took 11 s where both resident took 16-20. A render
  straight after a render leaves the model away rather than load it only to
  unload it. One "make a picture" request went from 120 s to 72 s, 55 s to 17 s
  of it between the finished picture and the answer.
- **A picture ComfyUI made is checked, not reviewed** (`AppSpec.makes_pictures`,
  `Vision.check`). Asked to "name concrete visual defects", the vision model
  always found some - steam from a fox's mouth, fur "not red enough" - and the
  9B redrew the picture for each, twice in one request, against its own
  briefing's one-render rule; one of those edits took 410 s. `CHECK` asks what
  the picture shows and whether it matches the brief or is plainly not what
  was asked, and forbids a list of small flaws; it writes 16-24 tokens where
  the review wrote 88-301. Frames of a project (After Effects, Photoshop, …)
  keep `REVIEW`: there the flaws are the work.
- **A photo finish is a second run** (`finish_edit`). In the edit's own graph
  ComfyUI kept the 19.5 GB edit model on the card while it staged Z-Image's
  11.7 GB beside it: "Model Initializing" for minutes, then 7 s a step, 410 s
  for one edit. Now the edit ends in a `PreviewImage`, the GPU is freed, and the
  finish run loads that preview (`LoadImage` with `"<name> [temp]"`): 119 s for
  the cold edit plus 31 s for the finish.
- **A face swap is crop, edit and stitch, in three runs** (`t_face_swap`). There is
  no face-swap model on the LLM PC (no InsightFace, ReActor or IP-Adapter), and
  `comfy_edit_image` on the whole picture gave the user strangers: in a 1 MP frame
  a face is a few hundred pixels of the edit model's attention. So SAM3 finds the
  faces (`find_faces`, a second or two; `face:8` asks for up to eight, and faces
  under a quarter of the largest are the crowd), each scene face is cut out as a
  square 2.4x its size and edited at 1024 px against the matching face (left to
  right, or `order`), the edits end in previews, the GPU is freed, and a SAM3 run
  masks the head before and after, grows and blurs the mask, fades it off the
  crop's edge and composites each head back where it was cut. Nothing outside
  the heads changes. SAM3's boxes come back through `PreviewAny`, whose text is
  in `/history` - the one way a graph's non-image values reach the bridge.
- **Every face is redrawn at full size** (`face_detail`, `redraw_faces`). A face a
  tenth of the frame's height is ~60 px of Z-Image's 1024 px sample, and it drew
  them that small: a wedding party of eight all came back smooth and waxy, eyes and
  teeth smeared - the detail pass enlarges that, it does not fix it. So with SAM3
  installed `comfy_generate` runs the picture to a preview with SAM3's face boxes
  beside it, then a second run crops each face `FACE_PAD` (2x) its size, enlarges it
  to 1024, resamples it with the same model at `FACE_DENOISE` (0.45) under
  `FACE_PROMPT` (the scene's prompt inside a face-specific one) and blends it back
  through a soft oval (`oval_png`, a greyscale PNG the bridge draws and uploads).
  Two details matter. Each crop is taken from the picture as composited so far, so a
  neighbour's crop never pastes an old face back. And the blend is the oval, not the
  crop's square: in a row of faces each square holds the next person's face, and a
  square blend ghosted it. A crop already 1024 px or larger is left alone, as is a
  face under `FACE_MIN`. Z-Image stays on the card between the runs. Measured on
  the LLM PC with the card free: 8 faces added 100 s (7 s sampling and ~5 s of VAE
  and model staging each) to a 68 s picture; one face adds ~15 s. `comfy_face_swap`
  ends with the same pass at `SWAP_DENOISE` (0.3) when Z-Image is installed, because
  the edit model's face read as pasted onto an old photograph; at 0.3 it takes the
  photograph's grain and light and keeps the likeness. `face_detail` false turns
  either off. The pass does not fix skin the prompt asked for: "weathered, ruddy,
  deep lines, visible pores" came back crackled and blotchy at every setting, with
  the pass and without, so the briefing says to name skin once and plainly.
- **Every picture a bridge saved says where** (`_meta.path` on the image block).
  The transcript's preview is shrunk to fit; right-click on it saves, opens,
  shows or copies the full-size file, and a double-click opens it
  (`Chat._picture_menu`). A picture with no file - a screenshot sent as data -
  is saved from its bytes.
- **The text encoder runs on the CPU** (`ENCODER_ON_CPU`, `device: cpu` on the
  `CLIPLoader`). It runs once a picture; the diffusion model runs every step. On
  the GPU the 8 GB encoder stayed resident beside the 12 GB Z-Image (or the
  19.5 GB edit model) and crowded it: with the whole card free, a first picture
  took 158 s with the encoder on the GPU and 49 s with it on the CPU, from the LLM
  PC's 128 GB of RAM. `COMFYUI_ENCODER_ON_GPU=1` puts it back, for a bigger card.
- **The detail pass encodes and decodes in tiles** (`VAEEncodeTiled`,
  `VAEDecodeTiled`, `TILES`): a whole 2-4 MP frame through the VAE beside
  resident models ran ComfyUI out of VRAM, and its fallback stalled for minutes.
- **One render per request.** Handed the vision review of its picture, the 9B
  rendered the same truck seven times over nitpicks ("does not look old enough"),
  34 minutes for one request. The briefing now says render once, show it, offer
  the change, and render again only when the picture is plainly not what was
  asked - and the briefing alone did not hold; the check above is what does. A tab
  whose model was unloaded - by this, or by hand in LM Studio - is refitted
  before its next turn (`Chat._reload_if_unloaded`), because the host's own
  just-in-time load is 8,192 tokens and truncates the briefing. Each result still
  names a starved GPU (`vram_note`, under `LOW_VRAM`) and says how long it took.
- **A silent ComfyUI is busy, not gone.** While it stages a model its HTTP server stops
  answering for 30 s or more. `_open` raises `Unreachable` for no answer at all, and
  `wait_for` keeps polling through it until its deadline, reporting progress as
  "busy". Before this, a render in progress failed as "cannot reach ComfyUI".
- **Progress keeps an MCP call alive.** `MCPClient._request` gave every call 180 s,
  and a four-minute render was cancelled at three, leaving the model to find
  `comfy_wait`. A `notifications/progress` on the call's own token now restarts its
  timeout, up to `PROGRESS_CAP` (30 min); a bridge that says nothing still times out.
  The bridge's own wait defaults to `DEFAULT_WAIT` (15 min).

### The Image Studio: a form over several ComfyUIs

The ComfyUI tab is a conversation; the Image Studio (`IMAGE_STUDIO`, an `ImagesSpec`,
a `PanelSpec` with `images = True`) is a form. It holds no other program's window,
so `_ensure` just marks it ready and calls `ImageStudio.start()`, the first thing in
it that touches the network (a tab is built before it is looked at, and the GUI
tests build every tab). `studio_imagegen.py` is the engine, with no tkinter;
`studio_images_ui.py` is the tab, a collaborator the window lends `_skin`, `_button`,
`_entry`, `_spawn`, `q` and `_animate`. Worker threads post `("images", sid, ...)`
events, and `_panel_event` hands them to `ImageStudio.handle`. The rules:

- **Backends are independent workers.** Each is a record in
  `image-studio/backends.json` (beside the settings): URL, WebSocket URL, enabled,
  roles, notes, and three flags. `shares_llm_gpu` means clear LM Studio off the card
  first (`Chat._images_make_room`, which is `_make_room`'s rule). `release_vram` means
  `/free` when that backend's queue empties. `encoder_on_cpu` does what it does in
  the bridge. VRAM is never pooled and no path is assumed to exist on the other
  machine: pictures reach a backend by upload (named by content hash, sent once),
  models by a filename that backend has. The defaults are the 5090 on this PC
  (`IMAGE_STUDIO_5090_URL`) and the 3090 (`COMFYUI_URL`).
- **Every ComfyUI call is in `ComfyUIClient`**, one per backend. Progress comes over
  the WebSocket (`studio_milanote.WebSocket`, read on a thread of its own so a timeout
  never cuts a frame). The *end* of a job is read from `/history` every two seconds
  regardless, so a socket that never opens costs the step counter, nothing else.
  `cancel_job` interrupts only its own prompt (`/interrupt` with `prompt_id`, or
  `/queue delete`): a bare interrupt would stop the chat tab's render.
- **Progress is ComfyUI's own events, shown as Queued → Loading → Sampling →
  Decoding → Complete.** The job opens the socket (`watch()`) *before* it posts
  `/prompt`, or the first `executing` events are gone. A node's stage comes from
  the template's `stages`, else its class (`stage_of`). A sampler that has sent
  no step yet shows as Loading, because ComfyUI moves the model onto the card
  inside the sampler. Steps show as "step 12 of 20 · 60% · node 40 KSampler",
  and a queued job shows how many prompts are ahead of it. A socket that will not
  open is said once (`Watch.error`, into the record's notes), and 120 s with no
  event (`QUIET_AFTER`) says which node it was on. That second one is real: a
  16384² FLUX job ran out of memory in the VAE, ComfyUI's worker died handling
  it, and the prompt stayed "running" with the GPU idle until ComfyUI was
  restarted. `compose` now refuses a size over the backend's `max_megapixels`.
- **Errors name the thing.** `explain()` turns ComfyUI's 400 into "node 1
  (UNETLoader): Value not in list - unet_name 'flux1-dev.safetensors' is not
  there (it has 3 others)" rather than the server's whole file list;
  `run_errors()` gives the node, class, exception and message of a failed run.
  Nothing is dropped quietly: an unreachable local URL says no ComfyUI is
  running on this PC (Studio Assist is not one) and how to start it (the
  backend's `start` field), and something answering on the port without a
  `comfyui_version` is reported as not ComfyUI.
- **What a backend has is read, not assumed.** A full check reads `/models/<kind>`
  and `/object_info` (refreshed each time). `missing_for()` and `lacks()` list
  every file the model's workflow needs that is not there, by exact name and
  `ComfyUI/models/<folder>`, plus every node class it lacks. The model menu
  ("ready on 5090 Workstation"), the Models window's status panel, routing
  (`has_model`) and `compose`'s errors all ask that one function.
  `python studio_imagegen.py --probe` checks `/system_stats`, `/object_info`,
  `/prompt` (an empty graph, which ComfyUI refuses without running anything),
  `/history` and the WebSocket on every backend, then prints each model's
  readiness.
- **Workflows are files.** `comfy_workflows/<id>.json` is an API-format graph plus:
  `{{placeholders}}` (a whole value keeps its type), `_when`/`_unless` nodes,
  `switches` (a link chosen by a value), `lora_chain` (where LoRAs hang;
  `{{model_out}}`/`{{clip_out}}` are its end), `references` (which reference kind
  feeds which image input), `files` and `needs` (what must be on the backend), and
  `stages.refining`. `fill()` is the whole adapter and refuses an unfilled value or a
  dangling link by name. Check a new template against the server's `/object_info`
  (the tests hold the shipped ones to filling; a live check is by hand).
- **Logical names.** A model (`models.json`) has `values` for every machine and a
  `backends` entry per machine that overrides them: other filenames, fp8, even
  another family or workflow, or `null` for "not installed there". A LoRA has `file`
  and `files` per backend. Compatibility is decided on the family *resolved for the
  backend the job lands on*: an incompatible LoRA is left out with a warning, and a
  LoRA of unknown family is applied with one. Nothing is dropped silently.
- **Identity and style are separate records.** An identity is a LoRA, a trigger, a
  strength and reference photos (copied under `image-studio/references/`). A style
  is a LoRA and/or prompt additions plus look defaults. The precedence is model
  defaults < style < preset < the form. `compose()` builds the prompt as triggers,
  then the scene, then the style. It is pure and does no I/O, which is how the form
  shows warnings before Generate.
- **A reference is typed** (face, pose, composition, style, source) and used only
  where the workflow declares that kind. Otherwise it is a warning, not a silent
  reuse. As of 2026-09-25 `flux_hq` takes `source` (image to image) and
  `style`/`composition` (Redux, when its two files are on the backend). Nothing on
  either machine does face or pose conditioning yet (no PuLID, IP-Adapter or
  ControlNet), so the likeness is the identity LoRA's.
- **One lane per backend, one job per picture.** `JobQueue` runs a thread per
  backend, so the two GPUs work at once. A batch of N is N jobs with seeds s..s+N-1,
  spread over every capable backend, so each picture's record states its exact seed.
  Auto routing (`plan_route`, over `route`) prefers backends whose roles include
  the preset's role, so FLUX goes to the 5090 when both could take it. It falls
  back to any enabled one that is up and has *every* file and node the workflow
  needs, and says why when nothing can take the job. The form shows the choice
  and its reason ("Will run on 5090 Workstation: preferred for Standard; …")
  before Generate, and Generate refuses, in words, what would fail there.
- **History is the record.** `image-studio/history/<date>/<id>.json` sits beside the
  pictures and holds the settings as submitted, the resolved prompt, model file,
  LoRAs with strengths, identities, style, backend, sampler values, the refine pass,
  warnings, timing and the submitted graph, plus every model file and the
  workflow. Each job's output is named `ImageStudio/<workflow>_<job id>`.
  Reuse Settings loads `settings` back. **Generate Again makes the same
  picture** (`again(record)`): the recorded seed, the sampler values it resolved
  to, and the backend it ran on when that can still take it (`prefer_backend`;
  another GPU may differ slightly, and the route line says so). New Seed is
  the variation. Measured on the 5090: the pixels match exactly, but the PNG
  bytes do not, because ComfyUI embeds the graph, and the output name in it
  differs per job.
- **FLUX runs the baseline first** (`flux_dev_baseline.json`): ComfyUI's own
  `flux_dev_full_text_to_image` recipe plus FluxGuidance, taking prompt, seed,
  size, steps, guidance, sampler, scheduler and output name. It has no LoRAs,
  references, refine or face pass. `compose` leaves any of those out *with a
  warning*. `flux_hq.json` (the layered graph) stays on disk for later, and
  the tests keep its LoRA/reference logic covered through a `flux-hq` model of
  their own. Layer features back one at a time, against the baseline.
- **The 5090's ComfyUI (2026-09-25)** is ComfyUI v0.37.2 (the 3090's version),
  git-cloned into `D:\ComfyUI` with its own Python 3.12 venv and PyTorch
  cu128 (Blackwell). The models live in `D:\ComfyUI-models`
  (`extra_model_paths.yaml`), so a reinstall keeps them. It is started by
  `D:\ComfyUI\Start ComfyUI (Image Studio).cmd` on 127.0.0.1:8188, and the app
  does not start it. FLUX files: `flux1-dev.safetensors` (Comfy-Org mirror, the
  same file as BFL's), `clip_l`, `t5xxl_fp16`, `ae`. Measured: 1024², 20 steps,
  ~2.6 it/s, 11-12 s end to end including the model load. The 3090 has no FLUX
  files (its entry expects `t5xxl_fp8_e4m3fn_scaled` and fp8 weights), so Auto
  never sends FLUX there. Z-Image Turbo at 1024² took 18-30 s there.

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

### The tab that holds a window: `PanelSpec`

Milanote is a web app with no public API and no MCP server, so its tab has no model,
no bridge, no transcript and no composer. `PanelSpec` (`panel = True`, `drivable`,
`bridged` and `research` all False) is in `TABS` but, like chat, not in `APPS`.
`studio_milanote.py` does the work:

- **The window is a browser we start, re-parented into the tab.** `Browser.start()`
  runs Chrome (else Edge; `STUDIO_MILANOTE_BROWSER` overrides) with `--app=` and a
  profile of its own (`STUDIO_MILANOTE_PROFILE`, default
  `%LOCALAPPDATA%\StudioAssistant\milanote-browser`), through `procs.spawn`. It finds
  the window by the pid, and `adopt()` makes it a `WS_CHILD` of the tab's frame. It
  must be its own profile. With the user's everyday one, a Chrome that is already
  running takes the launch over and there is no process of ours to find a window for.
  Chrome also refuses remote debugging on the default profile.
- **The window's own title bar is clipped, not removed.** An `--app` window draws its
  caption itself, so window styles cannot take it off. `measure()` reads the page's
  `outerWidth - innerWidth` and `outerHeight - innerHeight` over DevTools, and `fit()`
  places the window that far up and left of the frame, which clips the caption and
  the resize borders. Measure after `embed`, because the borders change once the
  window is a child: 0 px before and 10 px after at 150%. `fit()` does nothing before
  `embed`, because it would move a top-level window to the desktop's corner. The app
  is `SetProcessDpiAwareness(1)`. A DPI-unaware host put the child at the wrong offset.
- **Upload is a drop.** `Input.dispatchDragEvent` (`dragEnter`, `dragOver`, `drop`)
  with the file paths, at the middle of the page. Milanote makes a card of each file.
  A test page received both files with their full contents. DevTools listens on a port
  Chrome picks itself (`--remote-debugging-port=0`, loopback only) and writes to
  `DevToolsActivePort` in the profile. The WebSocket client is a stdlib one in the
  module. On the sign-in page an upload says to sign in rather than dropping.
- **Closing detaches first.** `_close_tab` calls `release()` (hide, `SetParent(None)`,
  `WM_CLOSE`) before it destroys the frame. Destroying a parent destroys its children,
  and Chrome would lose its window under it and be killed instead of closed.
  `Session.close()` then waits `CLOSE_GRACE` and ends the job.
- **In the GUI** `_build_transcript` hands a panel to `_build_panel`. `_select` hides
  the composer for a panel and puts it back (`before=self.strip`) for a conversation.
  `_ensure` spawns `_open_panel`, whose `("panel", sid, ...)` events `_panel_event`
  handles. Every other event for a panel tab, a `sid` of None included, is said on the
  panel's note line, because there is no transcript to write into. New chat, History,
  Send, Start and Capabilities do nothing on a panel. `tests/test_milanote.py` holds
  the DevTools client against a fake on a real socket, and the tab against a stub
  browser.

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
method names against `tkinter.Misc` / `tkinter.Tk` — a test does it for you. The one
exception is `report_callback_exception`, which Tk *means* to be overridden; the test
keeps a `DELIBERATE_TK_OVERRIDES` allowlist, and a second test holds every name in it
to being a real Tk attribute the class really defines, so the list cannot become a
hiding place for an accident.

**The queue pump must never die, and nothing may fail in silence.** `_drain` is the
only path from the worker threads to the UI, and `_handle` — which it calls — touches
widgets, images and transcripts. Anything `_handle` raised used to escape past the
`self.after(40, self._drain)` at the end, so the pump stopped *permanently*: every tab
went quiet at once (no tokens, no status, no idle, Stop stuck on) while the threads
kept filling a queue nobody read. The shortcut starts the app with `pythonw.exe`, which
has no stderr, so the traceback went nowhere and the app looked hung rather than
broken. Now the reschedule is in a `finally`, each event is handled in its own
`try`, and a failure goes to `_report`: the error log, plus a line in the transcript
naming the log. `report_callback_exception` routes a failing menu command or button to
the same place, for the same reason. Two rules follow — **nothing in `_report` may
raise** (it is reached precisely when a widget is already unhappy, so every step is
wrapped), and **the pump is one timer however it is entered**: `_drain` stands the old
tick down before arming the next, because the twenty-odd hand-called `_drain()`s in
`tests/test_agent.py` each used to arm one beside the live one, and Tk named every
orphan on the way out. `_quit` sets `closing` and cancels `drain_timer` and
`host_timer` through `_stand_down`; the GUI tests tear down with `_quit()`, not
`destroy()`.

**The modules pulled out of `studio_chat`, and the rules that keep them out.**
`studio_doctor` (where things are kept, the error log's one writer, the diagnostics
report), `studio_files` (attachments: headers, folder listings, the container copy)
and `studio_ui` (palette roles, `blend`/`rounded`/`clip`/`pretty_host`, `Pill`).
`studio_chat` re-exports every name it used to define, so the rest of the app reaches
for them where it always did — but edit them in their own module.
`tests/test_doctor.py::ModuleBoundaryTest` enforces the two rules worth having: the
headless pair must import with tkinter entirely unavailable (tested by blocking it on
`sys.meta_path`, not by reading the source — a *guarded* probe inside a function is
fine and wanted, since `python_rows` reports a missing tkinter on purpose), and none
of the three may import `studio_chat` back. `studio_ui` is exempt from the first:
`Pill` is a Canvas.

**What is left of `studio_chat` is one class, and that is the real shape of it.**
`Chat` is ~220 methods over ~3,700 lines. A mixin carve-up — `class Chat(SidebarMixin,
TranscriptMixin, …)` — would scatter the text across files while every piece still
reached into `self` state defined somewhere else, and would cost the one thing the
single file currently buys: everything about the window is findable in one place.
Do not do that. Further splitting should be real collaborator extraction, one at a
time, each owning its own state and reached through a narrow interface — the
animation engine (`_animate`/`_arm_anim`/`_anim_tick` over `anim`/`anim_timer`/
`anim_frame`) is the cleanest candidate and would make an `Animator` that takes a
widget to schedule on and a `report` callback for failures.

**`studio_doctor.py` must not import tkinter.** It holds the layout of
`%APPDATA%\StudioAssistant` (`settings_path`, `data_dir`, `error_log_path`,
`tasks_dir`), the one error-log writer, and the diagnostics report — and the whole
point of `python studio_chat.py --doctor` is that it answers "I clicked the shortcut
and nothing happened", which includes a Python whose tkinter is broken or missing.
Importing the GUI to ask what is wrong would fail for the reason being asked about.
`studio_chat` re-exports `settings_path`, `error_log_path` and `log_error`, so the
rest of the app still reaches for them where it always did; edit them in
`studio_doctor`. `--doctor` is handled before the DPI call and before
`claim_single_instance`, so it runs beside a copy that is already up, and its exit
code is `worst()`: 0 fine, 1 worth a look, 2 broken.

**The report is one function, rendered twice.** `doctor.report()` returns
`[(section, [(label, value, role)])]` where the role is a palette name; `as_text`
prints it with a mark per role for the console, and `Chat._paint_diagnostics` paints
it with a tag per role for Help > Diagnostics. Two renderings, one set of facts, so
the window and the console cannot drift. The probe does network I/O — a tailnet
timeout is seconds — so the window opens saying "Checking…" and the report arrives on
the queue as a `diagnostics` event, handled beside `icon` because it belongs to the
installation rather than to a tab and must still land with every tab closed.
`_paint_diagnostics` returns quietly when the window has been shut under it. Tag
colours are copied out of the palette, so `_diag_tags` is called again from `_theme`,
exactly like `_tool_tags`.

**`Session.prefix_tokens` and `Session.window` are kept, not just used.** The warm-up
measures the exact prefix (`usage.prompt_tokens`) and `_fit` knows the window the
model was loaded with; the app acted on both and then forgot them, so the first thing
anyone needs when a tab talks instead of calling a tool — prefix against window — was
only ever visible in a note that had scrolled away. Diagnostics reports them per tab,
and a tab with under `ROOM` left reads as an error, not a note.

**Saved tasks are the user's work; the app never removes one by itself.** Every
message checkpoints its tab into `tasks/<app>/<task-id>.json`, so that folder is the
app's memory of what has been asked of it — and it was reachable only through a file
dialog pointed at 32-character hex names, which is why nothing was ever resumed from
it. The header's **History** button (also File > Chat history…, Ctrl+H) lists the
active tab's past conversations newest first through `TaskRecord.summaries()`:
what was asked (the first brief), when, how many steps, the recorded status. A file
that will not parse is listed carrying its `problem` and offered no resume button
rather than being hidden — silently dropping it is how someone comes to believe the
app threw their work away. The conversation already on screen is filtered out, since
resuming it would replace it with a checkpoint of itself. There is **no automatic
pruning**: not on a timer, not to keep the folder tidy. Deleting is a per-task button
that asks first. (Nineteen saved tasks were examined when this was written; every one
held a real request. An age or count based sweep would have deleted work.)

**The launchers must not name a Python by its install path.** `Studio Assist.cmd`
pointed at `...\Programs\Python\Python312\pythonw.exe`, which is one Python upgrade
away from a shortcut that does nothing at all when clicked — no window and no error,
because `pythonw.exe` has no console to complain in. Both `.cmd` files now resolve
`pyw`/`py` (the Windows Python launcher, in System32, which survives a version change)
and fall back to `pythonw`/`python` on the PATH, and say what to install when neither
is there.

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
It is not: a full tool-schema set took about a minute to prefill cold - seconds, since
the model loads onto an empty card (see *A model loads onto an empty card*). The warm-up
pays that against *the exact prompt prefix a real message uses*, so the first question
returns in seconds. LM Studio's prefix cache survives across processes, which is why
this works at all. Each tab has its own prefix and so warms up separately, the first
time it is opened.

**The window the model is loaded with is this app's to set, and it sets it.** LM
Studio loads a model at its default, 8,192 for most, and every tab's fixed prefix -
briefing plus tool schemas - is most or all of that: 7,707 tokens measured on the
ComfyUI tab before a word is said, ~11,000 estimated for Resolve, ~20,000 for After
Effects (65 tools). Under that the host truncates the request and the model loses its
own tool results: the 30B called `list_folder` ten times running with the same
arguments until the step limit, and a thinking model was cut off with
`finish_reason: length` (which `LLM.stream` explains in these terms). For a while the
warm-up only *reported* this - `headroom_note()`, with the `lms load --context-length`
to run on the LLM PC - on the grounds that the GPU is on another machine. That was the
hurdle: walk to the other PC, eject, reload, and LM Studio's next just-in-time load is
8,192 again. The app already loads the vision model through `/api/v1/models/load`; the
same call takes `context_length`, so `fit_model()` loads, or unloads and reloads, the
executing model with `wanted_context()` - a power of two, 16,384 at least, `ROOM`
(8,192) past the prefix, never past the model's `max_context_length`. `_boot_session`
fits twice: before the warm-up from `estimate_tokens()` (chars / 3; tool JSON measures
~3.5) so the model is not first loaded just in time at the default and thrown away, then
on the warm-up's exact `usage.prompt_tokens`, and a reload there warms up once more,
because a reload empties the host's prefix cache. Two facts about LM Studio shape the
code: loading a model that is loaded makes a *second instance* (`<model>:2`) rather
than reloading, so `loaded_instances()` (from `/api/v1/models`) is unloaded first,
every one; and a load beside another tab's request cuts that request off, so `_fit`
gives the old advice instead while any other session is `busy`, and `fit_lock`
serialises two tabs booting at once (the second finds the first's load and does
nothing). A host with no REST API says nothing, and nothing is fitted. `headroom_note()`
remains the fallback when a load fails or the model is already at its maximum.

**A model loads onto an empty card, and the vision model goes on after it.** LM
Studio gives the GPU to the model it loads first; one it loads beside another is
placed partly in system memory, and stays there for as long as it is loaded. Measured
on the LLM PC's 24 GB with the 30B at 32k and the 7B vision model (28 GB between
them): vision model first, the 30B decoded at 20-31 tokens a second with a 154 tok/s
cold prefill - and stayed that slow after the vision model was unloaded; the 30B first,
it decoded at 77-79 (410 tok/s prefill with the vision model beside it, 2,600 alone).
The window used to load the vision model the moment the probe answered, ahead of every
tab, so every app tab ran its 30B at a third of its speed. Now `fit_model(keep=...)`
unloads everything but `keep` before a load (`_fit` passes `()`: the only way it gets
there is with no other tab busy), `_boot_host` loads nothing, and `_boot_session` spawns
`_load_vision` once its own model is on - not beside a tab with `gpu_tools`, which
clears the card for every picture anyway. A load the fit makes takes the vision model
with it, and `_fit` marks it `needs_load` again. The 30B and a vision model still do
not fit in 24 GB together; a pair that does (a smaller executing model, or one that
sees for itself) would get the prefill back too.

**Never test the GUI with synthetic keystrokes.** `SendKeys` types into whatever
window has focus, not the one you meant. It has already leaked a test sentence into
the user's chat window mid-run. Test by constructing `Chat()` in-process and calling
its handlers, and assert on widget geometry for layout. Screen-capturing the window
to look at it is fine — that's read-only.

**The app is called Studio Assist; the paths on disk are not, and must not be.**
`APP_NAME` is the one place the displayed name lives — title bar, About, the "already
running" box, the crash box. It is deliberately *not* in the header: Windows already
shows the title on the taskbar and in Alt-Tab, and repeating it a line below bought
nothing. The header's job is to say what the tab you are looking at is doing, so the
status starts that row. A test asserts the name appears there no more. Four things deliberately keep the older spelling
because they are identities rather than labels, and renaming them would orphan what is
already written under them: `%LOCALAPPDATA%\StudioAssistant\` (settings, lessons, task
records, the OpenCode workspace), `studio_assistant_error.log`, `studio-assistant.ico`,
and the CEP panel's `ExtensionBundleId`. The panel's *display* name did change, so
`python studio_premiere_mcp.py --install-panel` has to be re-run for the entry under
*Window > Extensions* to read "Studio Assist Bridge"; until then the app's instructions
name a menu item the installed panel does not have. The launcher is `Studio Assist.cmd`
now, so a shortcut pointing at the old filename needs re-pointing.

**Everything that moves runs off one tick, and only while something is happening.**
`ANIM_MS`, `Chat.anim` (key → `draw(frame)`) and `_anim_tick` are the animator; nothing
else may call `after` to animate. The reasoning is the queue pump's, plus one more:

- *One timer.* A timer per animation is a timer per orphan on the way out, and Tk
  names every one of them. `_anim_tick` stands the old tick down before arming the
  next, because a hand-called tick — the tests are full of them — otherwise arms one
  beside the live one. It respects `closing`, and `_quit` clears `anim` and stands
  `anim_timer` down with the rest.
- *Nothing escapes a frame.* `_draw_once` guards every paint, from the tick and from
  `_animate`'s immediate first paint alike — that one matters more, because `_animate`
  is called from `_apply_status` and friends, which run on every event, so an animation
  that raised on registration took the event with it. A `tk.TclError` is a widget
  destroyed under its own animation: ordinary, silent, dropped. Anything else goes to
  `_report`.
- *Nothing animates that is not happening.* The tick is armed only while `anim` has
  something in it, so a settled window draws nothing — and `draw` returning `False`
  drops it. Every animation must therefore be tied to a real in-progress state and be
  able to end. Two were caught being unable to: the arc sweeping while any bridge was
  unstarted (a bridge that never starts is a permanent animation), and the rail's dot
  keyed on the bridge role rather than `Session.booting` (the host coming back resets
  every stuck tab, and the ones you are not looking at wait to be selected — settled,
  not busy). `test_a_settled_window_animates_nothing` is the guard; keep it passing.

**A trailing ellipsis is the marker for "still happening".** A status, a button label
or a caption ending in `ELLIPSIS` is one describing work in progress, and `_ellipsis`
animates the dots; anything else is shown once and its animation dropped. The author
writes `"working" + ELLIPSIS` at the point where they know that, and nothing else has
to be told — no fourth element on the status tuple, no list of phrases to keep in sync.
The literal character never reaches a label. New status for something in flight: end it
with `ELLIPSIS` and it animates for free.

**`blend()` builds both channel lists eagerly, and the reason is not style.** A
generator expression per colour reads better and is wrong: it closes over the
comprehension's loop variable, so by the time `zip` draws from either one both yield
the *second* colour. Every blend in the window then silently came out as its second
argument — the dots stopped pulsing, the placeholder's sheen vanished and the arc's
ghost went the colour of the panel, with no error anywhere. A test pins the midpoints.

**Drawn canvases are repainted on a theme switch, never reconfigured** — `arcs` joins
`dot_role` and `marks` in `_forget` and `_theme` for exactly the reason those exist. An
animation is the exception that needs no hook: its `draw` reads `self.C` live, so it
picks the new palette up on its next frame by itself.

**Surfaces are told apart by contrast and spacing, not by 1px rules.** There are no
hairlines under the header, beside the rail, under the tab strip or above the
connections; the light theme is a `#f7f7f8` canvas with white surfaces on it. A field
or the composer sits on the canvas raised by `ui.lifted`'s two-pixel shadow, and a
field draws the accent ring only while focused. What keeps an edge: the tooltip (it
floats over anything) and the Preferences theme cards (the ring *is* the choice).

**The composer floats.** A white card 24px in from the sides and 20px off the bottom,
no footer band behind it: the text on top with a placeholder label (Tk's Text has none;
it is shown and hidden on `<<Modified>>`), and under it `+ Attach` and the folder
glyph on the left, the Enter hint and a round accent send button on the right. The
button's idle label is `SEND` (an arrow); busy it still reads `Stop` / `Stopping…`, and
`Pill(round=True)` grows from a disc into a lozenge to fit. It draws ovals, not
`rounded()`: a smoothed polygon asked for half-height corners undershoots badly.

**Nothing in this window is square, and Tk cannot bend a widget.** A Frame, a
Button and an Entry are rectangles; the only thing that curves is a smoothed polygon
on a Canvas. So every rounded surface in the app is the same construction — a canvas
that draws the shape, with the real widgets in a frame placed on top of it through
`create_window`, and a `paint()` the canvas's or the frame's `<Configure>` calls:
the composer's surface, a tab chip (`_make_tab`), a rail row's hover (`_app_row`), an
attachment chip, the question form's card, a text field (`_entry`), a Preferences
theme card. Three rules come with it:

- *The frame must clear the curve.* A corner of radius `r` bulges `r(1 - 1/√2)` ≈
  `0.29r` past the corner, so the inset has to beat that or the frame's square corners
  poke out of the shape and the whole thing looks broken rather than round. Each site
  names its own `PAD` and `R`; the tab's are `_metrics`' `tab_pad` / `tab_r`, because
  `_fit_tabs` has to measure against the same number.
- *The inset is height, and height adds up.* Nine rail rows at four pixels a side
  pushed the last one off the rail. Three is enough, and the gap it leaves between
  rows replaced the `pady` they used to be packed with.
- *It has to be repainted on a theme switch.* `_theme` reconfigures widgets, which
  does nothing for a colour plotted into a canvas item. Tab chips go through
  `_paint_tab`, rail rows through `_build_apps`; everything else registers with
  `_repaint_on_theme`, swept by widget in `_forget` because chips are rebuilt on every
  attach and every send. A picture's rounded corners are the sharp case: they are the
  background painted over the image (Tk will not clip a PhotoImage to a shape and it
  has no alpha to mask with), so a switch without a repaint leaves four blots of the
  old palette on every picture in the transcript.

**Every button is a `Pill`, through `Chat._button`.** `PILL_ROLES` holds the four
kinds — `accent`, `quiet`, `option`, `ghost` — as the `(bg, fg, hover, disabled,
disabled fg)` tuple `Pill.paint` reads. A `Pill` is a canvas, so it is repainted from
`self.pills` rather than reconfigured, it answers `cget("text")`, `set(text=, state=)`
and `invoke()` the way the Button it replaced did, and `anchor="w"` makes it a
full-width row that reads from the left and wraps — that is what the question form's
options are. A test reads both source files and fails on a literal `tk.Button(`.

**The bridges' mark is drawn, not a glyph.** MDL2's chain link says "two things
fastened together", which is not what a bridge is or what that row reports, and a font
glyph cannot show a span going up. `_arc` plots it: shallow on purpose, because at 18×13
a tall one reads as a chevron, and the piers and abutments that would fix that only
thicken the middle — both were tried at size. `_arc_state` draws it across once when the
state changes and then settles.

## Processes: nothing outlives the app

Every subprocess the app keeps — an MCP bridge, a COM bridge's PowerShell worker — is
started with `studio_procs.spawn()`, never a bare `Popen`. Read the module docstring
before changing that; the short of it:

- **Windows does not kill a dead parent's children, and `Popen.kill()` ends one process,
  not a tree.** The After Effects bridge is four deep (`cmd /c npx` → `node npx-cli` →
  `cmd /c after-effects-mcp` → `node server.js`); killing the top left the `node` at the
  bottom running with nobody on its pipes. So did a crash, a Task Manager kill, or a test
  run stopped half way. Leftovers piled up across a day of development.
- **So each child gets its own job object** with `KILL_ON_JOB_CLOSE`. It is created
  suspended, put in the job, then resumed, so even a grandchild started in its first
  instant is inside. `Child.stop(grace)` closes stdin (a bridge's cue to exit), waits,
  then terminates the job — the whole tree. If this process dies any other way, the
  kernel closes the job handle and does the same. `tests/test_procs.py` kills a parent
  outright and checks the grandchild is gone.
- **Only what we started is touched.** Nothing is found or killed by name. The apps are
  not our children: `launch()` starts them detached with a plain `Popen`, on purpose,
  because After Effects must outlive the window; Photoshop and Illustrator are started
  by COM's own service. Neither is ever in a job of ours. The OpenCode container is left
  running on quit as documented under *The app in a box*.
- **Quitting is parallel and off-screen.** `_quit` withdraws the window, closes every
  tab's bridge at once with `QUIT_GRACE_S` between them, then `procs.stop_all(0)` for
  anything left. One tab at a time, each allowed seconds, was a frozen window. Ctrl+C,
  Ctrl+Break and SIGTERM go through the same `_quit` (`procs.on_shutdown`); in the CLI
  they raise `KeyboardInterrupt` so its `finally: mcp.close()` runs. `atexit` stops what
  is left on any interpreter exit.
- **One copy at a time** is `claim_single_instance()`: a loopback port (57733) held for
  the life of the process, released by the OS however it ends, so there is no stale lock
  file to clear after a crash.
- **`ComHost` makes its temp folder on first use and removes it on `close()`** (also run
  at exit). It used to make it in `__init__`, and each bridge module builds its host at
  import — every test run and every bridge start left a `studio_com_*` folder in `%TEMP%`.
- **Idle is cheap.** The event pump polls every `DRAIN_MS` while events flow or a tab is
  busy and every `DRAIN_IDLE_MS` otherwise; the animator's timer is armed only while
  something animates. Tk draws with GDI: the window uses no GPU.
- **The Premiere panel closes its server on `unload`**, so reloading the panel does not
  find 7787 held by the page it replaced. It has no timers; it costs nothing idle.

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
- **The host probe is re-runnable; nothing about the host is final.** `_boot_host()`
  is one `probe_models` call with an 8 s timeout, and a PC still waking or a server
  not yet started fails it once. It used to leave `self.llm` None for the life of the
  window, every tab at `no inference host` with no button, and reopening the window as
  the only way on. Now `_boot_host(first=False)` is **Connect**: the header button
  (`Session.host_down` makes `_apply_status()` label it `Connect` and `_on_fix()` send
  it to `_connect_host()`), the Inference row's menu (`_menu_host()`, built apart from
  its posting like the others so tests can read it) and Bridges ▸ Connect to the
  inference host. `host_booting` keeps two probes from running at once; `host_ready`
  still means only "the first probe is over", which is all `_boot_session` waits for.
  The probe ends by posting `host_probed` with whether it succeeded, and
  `_host_probed()` on the UI thread unsticks every `host_down` tab: the active one is
  `_ensure`d now, a ready one is ready again, the rest wait to be selected — boot stays
  lazy. A probe that fails after an earlier success leaves `self.llm` in place, and one
  that finds the same model keeps the same `LLM` object, so the tabs booted on it stay
  in step with the row (`_draft_check` compares by identity). Mid-conversation,
  `LLM._open` raises `eng.HostUnreachable` (a `RuntimeError`, so nothing that caught
  one before changes) for a `URLError`; `_turn` and the warm-up answer it with
  `_host_lost()` (the row red, `unreachable`) and a fixable `inference host
  unreachable` status, and the next request that gets through calls `_host_back()`,
  which restores the row the probe left in `host_ok`.
  **The window retries by itself.** The LLM PC drops off on a timer, so a button
  alone would be pressed every time. A failed probe, a tab booting with no model and
  a turn that lost the host all end in `_arm_retry()` (the worker paths post
  `host_retry`), which schedules one *quiet* probe with `after(HOST_RETRY_MS)`;
  `host_timer` keeps it to one pending at a time, `_retry_host()` gives way to a
  Connect already running and stops when `_host_wanted()` is false (a model, and no
  `host_down` tab). A quiet probe (`_boot_host(False, quiet=True)`) moves the row and
  the status and writes nothing — the transcript must not gain the same paragraph
  every half minute; success is said once, as for Connect. The paragraph itself comes
  from `_explain_unreachable()`, which asks `eng.host_alive()` — `tailscale ping` for
  a 100.64/10 address with the CLI on PATH, `None` otherwise — because Windows
  Firewall drops a port with no listener rather than refusing it, so "asleep" and "up
  with LM Studio's server stopped" are the same `timed out` to `probe_models`; the
  wording names which, and what to change on the LLM PC (no sleep timer; LM Studio's
  service on login). Loud probes only: the ping is a subprocess. Tests:
  `TestHostUnreachable`, `TestHostAlive`, and the `connect`, `keeps_trying` and
  `which_half` tests in `TestGui`, which keep the real `_boot_host` behind the
  fixture's stub as `real_boot_host` and cancel `host_timer` in `_restore_host`.
- **An empty tab shows its app, not suggestions.** `_build_hero()` places the app's
  mark (`marks_px["hero"]`, read from its .exe like the others) and name over the
  middle of the transcript; `_welcome()` shows it, and the first message, a
  reopened conversation or any `err` line hides it — an error must never sit
  under it. The registry's `examples` are no longer shown in the GUI.
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
- **The task record is the last message of every request, never the second.**
  `context_messages` appends it after the conversation. It changes every turn -
  `status` alone flips to "working" before each run - and LM Studio caches by
  prefix, so with the record at position 1 the cache matched only through the
  system prompt and tools and the whole history was prefilled again on every
  message. A test asserts that turn N's request, minus its record, is a prefix of
  turn N+1's. Anything else that changes per turn goes at the tail too.
- GUI task JSON files live beside settings in `tasks/<app>/<task-id>.json`. Persist
  intent before dispatch and results afterward. A disk failure stops edits. Task
  records preserve conversational progress, not project backups or undo state.
- Unknown write outcomes must not trigger blind repeats. Restored tasks need a
  project read before editing. Read-back guards do not independently prove visual
  or semantic correctness; distinguish observations from model claims.
- **The read-back reminder names the call, and the executor looks for itself
  when it can.** An `AppSpec` carries `readback` - `(read tool, [id argument
  names])` pairs, most specific first - and `review` - the read-only screenshot
  tool and the ids it needs. When the model says it is done with an unverified
  edit, `Executor._auto_review` takes the screenshot (ids from the last write,
  else the latest call that carried them) and appends the vision model's review
  as the next user message, journaled with `auto: true`; that clears the
  read-back obligation because something looked. It runs only with a vision
  model - without one the picture would clear the obligation while nobody saw
  it - and at most twice a run. Otherwise `_readback_hint` names the exact read
  with the write's own ids (`call get_layer_full with {"compId": 12, "layerId":
  40}`), because a 3B model copies an instruction it can copy and skips one it
  must translate. Both tables are checked against the recorded contracts and
  the bridge tool tables (`tests/test_mcp.py`, `TestVerificationHints`): a read
  the default groups do not expose, or an id the tool does not take, fails.
  Resolve has neither yet - its compound tools do not fit the id-argument shape.
- **A reply that announces the call instead of making it does not end the run.**
  "Now I'll generate the image" with no `tool_calls` used to be accepted as the
  answer whenever nothing had been written yet, so the ComfyUI tab twice promised
  a picture and stopped, journal empty, looking finished. `announces_work()`
  reads first-person future phrases (`PROMISE`) in a reply that is not a
  question; when the tab has tools and this run has called none, the executor
  appends `PROMISE_HINT` as the next user message once and lets the model act.
  A second promise ends the run with `Nothing was done: ...` as the status and
  the `sys` line, so the user sees an empty run named rather than a plan. A
  sign-off after real work ("I'll be here if you want another seed") is not
  nudged - the count is per run - and `BASE_RULES` says the same thing in the
  prompt, for the models that read it. `tests/test_tasks.py`
  `TestPromisedWork` replays the transcript that found it.
- **An answer beside nothing but `studio_task_update` is the answer.** The chat
  tab's model wrote its reply and logged it in one step; asked for another step it
  wrote the same reply again, once or twice, and the user read it repeated. When
  every call in a step is the record update, it succeeded, the reply is non-empty,
  does not `announces_work()` and no read-back is owed, the run ends there with
  "response complete". A plan plus "I'll make the comp" still continues.
- **Stop interrupts the reply being streamed.** Cancellation used to be checked
  only between steps, so the rest of a reply kept arriving after Stop. The
  executor's token callback raises `Cancelled` once `cancel` is set; it unwinds
  through `LLM.stream`'s `with resp:`, closing the connection so LM Studio stops
  generating, and nothing the partial reply asked for runs. It lands on the next
  token: a Stop during prompt processing still waits for the first one.
- **The same read, with the same arguments, straight after itself, is not
  dispatched twice.** `_dispatch` counts the trailing journal entries with the call's
  `signature`: the second time it returns the first answer, marked, and journals a
  `repeat` entry (status ok, `verifies` false, so the count reaches three); the third
  raises `RepeatedCall`, which `_run` turns into a stop that names the cause - the
  model is not taking its results in - and skips the rest of the batch. Only reads:
  two `comfy_generate` calls with one prompt are two pictures, and a read after a
  write is a fresh look. The 30B at 8,192 was the case; the window fit above is the
  fix, and this is what stops the ten-call loop when something else causes it.
- `readonly()` decides which calls owe a read-back from the tool's name prefix or its
  MCP `readOnlyHint` annotation. A bridge written here must annotate its reads
  (`READ_ONLY` in every bridge here); unannotated, `comfy_status` counted as an
  edit and the model was nagged to "inspect" until it cycled on `studio_task_update`.
  A successful call that returns an image has done its own read-back — a generate
  that waited for its files is the observation, not something to inspect afterwards.
- Stop prevents subsequent dispatches; it cannot undo or guarantee cancellation of
  an in-flight operation. Complete tool-result envelopes when stopping a batch.
- **The executor settles before it asks the model anything** (`Executor._settle`,
  `eng.settle`): before every request, and when a run ends, so a reflection or the
  next turn never finds the model a render sent away still gone and has the host
  load it just in time at 8,192. Any other bridge has no `settle` and costs nothing.
- An `AppSpec` may carry `models`, small models it prefers, best first. ComfyUI did,
  because a 30B model resident beside a diffusion model on the same GPU is VRAM the
  pictures could have had - and that is what made the tab unusable: qwen3-1.7b under
  the ComfyUI briefing and nineteen tools described the generation it was about to
  make and never called `comfy_generate`, or called `comfy_upload_image` on a folder
  with a double-escaped path, then wrote its plan as a code block and asked "would you
  like me to". The shared 30B, given a window that fits, made the picture. ComfyUI
  sets `models` again, to `qwen3.5-9b-deepseek-v4-flash`: tested live on the
  ComfyUI briefing, it wrote a photographer's prompt and called `comfy_generate`
  on its own. With it, `gpu_tools` makes room for each render (see *What ComfyUI
  makes, and why it takes the time it does*). Try any smaller model the same way,
  with a real request through the CLI, before listing it. `STUDIO_MODEL_COMFYUI`
  still pins one. `AppSpec.model_for(ids,
  shared)` resolves it against what the host serves — `STUDIO_MODEL_<APP>` pin, then
  the list, then the shared model, never a model that is not served — and returns a
  note the tab prints once. The GUI keeps one `Session.llm` per tab, fixed at boot:
  the executor, both warm-ups and the host's cached prefix must agree, and tabs with
  no preference share the window's handle (`Chat._llm_for`). The CLI resolves the
  same way unless `--model` is given.
- **What is in VRAM is not what the user chose, once the app loads models of its
  own.** `pick_model` put "what is loaded" before the known-good list from the start,
  so a model loaded by hand in LM Studio was the one used. After the window began
  loading a vision model and LM Studio a draft, the first loaded id after an LLM-PC
  restart was `qwen2.5-vl-7b-instruct`, and every tab ran on it - a 7B that answers a
  tool call with advice to open File Explorer. `probe_models` now returns every
  loaded id, and `pick_model` goes: the pin, the default when it is in VRAM, anything
  else in VRAM that is not a helper (`helper_models()`: the preferred vision models
  and every draft in `DRAFT_MODELS`), the known-good list, the first served. A model
  loaded by hand still wins over a default that is not loaded.
- **Speculative decoding is a request field on some hosts and a load field on
  others.** Older LM Studio takes `draft_model` in the `/v1/chat/completions` body and
  loads the draft just in time; the studio's current one refuses that with "must be
  configured at load time, not prediction time" and takes it on `/api/v1/models/load`
  as `speculative_draft_model` with `speculative_draft_simple: true` (the field is
  named `draft_model` nowhere). It is not wired at load: a 1.7B draft ahead of a
  30B-A3B MoE, whose active parameters are 3B, is as likely to slow it as speed it,
  and nothing measured says otherwise. So on this host the first request per `LLM`
  pays one refused round trip, the note below is printed once, and the tab runs
  without; `STUDIO_DRAFT_MODEL=off` skips the round trip. `LLM(draft=...)` puts it on
  every request, streamed or not. `resolve_draft(model,
  ids)` picks it the way the vision model is picked: `STUDIO_DRAFT_MODEL` when served
  (`off` disables), else the smallest served model of the executing model's family
  from `DRAFT_MODELS`, else none — a draft has to share the main model's vocabulary,
  and `draft_family` matches a whole name (`qwen3` is not `qwen3.6`), never a prefix.
  A draft-sized model gets no draft. A tab with a model of its own resolves its own
  draft (`_llm_for`); `--draft` is the CLI's pin. **A pair the host refuses must not
  cost the tab:** `LLM._open` retries an HTTP error once without the draft, and only
  when that succeeds drops the draft for good and sets `draft_note`; the retry failing
  too means the error was not the draft's, and the original is raised with the draft
  kept. `_draft_check` says the note once, in the tab the request was made from, after
  each warm-up and turn. The Inference row names the draft and, when the host reports
  `accepted_draft_tokens_count`, the share kept (`LLM.drafted`, `_host_line`) —
  speculative decoding backfires when the draft is mostly wrong, and that is how to
  see it. Never pair across families by guessing: the host answers a mismatch with an
  HTTP error for the entire request.
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
  `/api/v1/models/load` — the GUI does it on a worker once the first tab's own
  model is on (never before it: see *A model loads onto an empty card*), the CLI
  just in time on the first picture; a host without that endpoint
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

## What the model learns, asks and looks up

Four mechanisms, all in service of one fact: the executing model is small and
starts every session knowing the app in general and nothing about this studio,
this bridge's failures, or what the user said last time.

### The research sidecar: files and the web on every tab

`studio_research_mcp.SERVER` — the Chat tab's bridge — is offered to every app tab
beside its own bridge. `AppSpec.research` is True for every entry (`ChatSpec` says
False: its bridge *is* the server); `_boot_bridge` and the CLI's `main()` make a
`Loopback` over it (`eng.research_client()`), append its tools after the bridge's
and its schemas with them, and put one `eng.Router` in front of both so the executor
still sees one client. Things that follow:

- **The sidecar's tools are not in `app.groups`.** `tool_names()` is the bridge's
  working set; `Session.offered(wanted)` is what the model gets — the bridge tools
  in `wanted` plus `s.sidecar_names` — and both `_boot_bridge` and the capabilities
  window build `s.tools` through it. Rebuild `s.tools` any other way and the tab
  silently loses the web. The tools window still lists the bridge alone.
- **`verification_read` says no to every sidecar tool.** A page or a brief on disk
  says nothing about whether an edit landed; without that, `fetch_page` after a
  write cleared the read-back obligation.
- **`AppSpec.docs` lists only pages the bridge can read.** `helpx.adobe.com` answers
  the bridge's browser user agent with 403, so Adobe's user guides are reached
  through `search_web` snippets only, and the prompt says so; the scripting guides
  on docsforadobe.dev, the Resolve API mirror, ComfyUI's docs and aereference.com
  all answer. `tests/test_lessons.py` refuses a helpx URL in `docs`. Verify a new
  URL with `fetch_page` before adding it: a 403 in the list teaches the model that
  looking things up does not work.
- The briefing is `AppSpec.briefing()`: the app's `craft` block (`CRAFT_EDITING`
  for Resolve and Premiere, `CRAFT_MOTION`, `CRAFT_DESIGN`, `CRAFT_IMAGES`), then
  `CREATIVE_RULES`, then `LOOKUP_RULES` with the docs folded in. Craft is written as
  rules the model can apply, not taste it is assumed to have.

### The notebook: lessons per app

`studio_lessons.Notebook` is `lessons/<app>.json` beside the settings, loaded by
`Chat._session()` and by the CLI, and best-effort in both directions like the tool
library. Four sources, marked on each lesson and ranked when the notebook is full:
`user` (a message that begins "remember" / "from now on", kept in the user's words
by `explicit_lesson`), `model` (`studio_remember`), `review` (the reflection) and
`error` (a validator refusal, learned deterministically from `Executor.refusals` —
a fact about the contract the model got wrong once). `eng.learn_from_run()` is the
one place a run teaches; it never raises.

- **The reflection runs only after trouble.** `Executor.trouble` is any TOOL ERROR
  in the run; `looks_like_correction(brief)` is the other trigger. A clean run makes
  no extra request. The reflection is one non-streaming `chat` with the whole
  exchanges that fit in `REFLECT_CHARS`, no tools, `max_tokens` capped, and its
  reply is **never appended to the conversation** — it must not cost the next turn.
  `parse_reflection` accepts only a reply carrying `Lesson:`; anything else is NONE.
- **Lessons are in the system prompt at boot and at the tail mid-session.**
  `Session.prompt()` builds `messages[0]` from `app.chat_prompt(studio, notebook.brief())`
  at `_session()`, on `reset()` and on a learned bridge; `brief()` marks what it
  carried, and `Executor._fresh_lessons()` puts the rest into `context_messages`'s
  tail (`extra=`) after the task record. Never rewrite `messages[0]` mid-way to add
  a lesson — the prefix cache, again. Every warm-up passes `s.messages[0]` itself,
  not a rebuilt prompt, for the same reason.
- `studio_ask` and `studio_remember` are `INTERNAL_TOOLS` with the task-record and
  tool-maker tools, in that order, before the made tools; `toolsmith.reserved()`
  already refuses `studio_` names. The Lessons window (`File > Lessons for this
  tab...`) is rebuilt on every forget; a forget while the tab is busy is ignored,
  because the worker may be writing the notebook.

### The studio brief

`studio.md` beside the settings (`eng.studio_brief_path()`, `read_studio_brief()`),
edited in `Chat._studio_window()`. `STUDIO_TEMPLATE` is what the editor opens with
when there is no file; saving it unchanged saves nothing, and `studio_section("")`
is empty, so no prompt ever carries the template's questions. The section is
bounded (`STUDIO_BRIEF_CHARS`) and goes after the quality rules and before the
lessons — the two parts that change between sessions come last. On save, a tab
whose conversation has not started (`len(messages) == 1`) takes the new brief at
once; the rest keep theirs until New chat, and the editor says how many.

### The question form: `studio_ask`

A question with two to five choices and an optional `multiple`. `Executor._call`
records it in `self.asked`, emits `("ask", payload)` and answers the model with an
instruction to stop; `_run` ends the run after that batch with status `response
complete; waiting for the user's answer`, so the GUI shows *ready* and the CLI
(`ask_at_terminal`) prints a numbered list and continues the same task with the
reply. In the GUI `_show_ask` embeds a frame in the transcript with
`window_create` — buttons, or Checkbuttons and a Send, and *Something else…* which
focuses the composer — and a click goes through `Chat._send`, the same path a typed
message takes; `_settle_ask` greys the form on either. Two things learned building
it: a frame embedded in a `Text` has no height until an idle pass, so `see("end")`
has to be repeated from `after_idle` or the form sits below the fold; and the
answer must go to the session that asked, not `cur()` — the user may have switched
tabs.

### The folded call rows

Every tool call is one row in the transcript: the tool's name (and its `action`,
for a compound tool), a red *failed* or *not run* when it went wrong, and behind a
click the call as the model made it - every argument whole, so a `run_jsx` /
`ps_run_jsx` script reads as the script - with the result under it.

- **`Executor._call` is the one place a call is announced.** It emits `("tool",
  {"name", "arguments", "via"})` before dispatch and `("tool_result", {"name",
  "text", "status"})` after - `status` is `ok`, `error` or, from `_skipped`, the
  `skipped` of a call the executor refused after a stop or three errors. Every
  route in goes through `_call`, so a made tool's steps (`via` naming it) and the
  auto-review's screenshot are announced like the model's own calls; a step's
  result arrives before the made tool's own. The CLI prints the same events
  clipped; the GUI keeps them whole (`CALL_TEXT_LIMIT` per value is the only cap -
  the task record has the rest).
- **In the GUI a row is text, not a widget.** `_show_call` writes the header under
  `tool` + `call_head` + `call:<n>` and the body under `tool_body` + `body:<n>`
  with `elide` on; `_show_result` finds the newest row of that name still in
  `Session.open_calls` (a stack, so nested steps pair up) and inserts the result
  at the end of that body tag. Tag bindings on `call_head` do the click and the
  hand cursor, and `_toggle_call` flips the elide and the chevron (`g["closed"]` /
  `g["open"]`, MDL2 by codepoint like the rest). Elided text is invisible but
  present: `view.get` returns it, `bbox` is `None` for it, and a `see("end")`
  after an expand near the fold is what keeps the header on screen.
- **Per-row tags go with the text.** `_clear_view` deletes `call:`, `body:`,
  `mark:`, `status:` and `stage:` tags, empties `open_calls`, and stands down the
  animations of every widget embedded in the transcript before the delete destroys
  them; use it, not a bare `delete("1.0", "end")`, wherever a transcript is emptied.
- **What the row is waiting for, it says.** The `status:<n>` range holds an embedded
  `_dots` canvas rather than a static `…`: a call that takes a minute used to look
  exactly like one that had hung. `_show_result` stands its animation down and
  deletes the range, which is also what destroys the widget. Embedded widgets go in
  through `_embed`, which tags the one character they occupy — a `window_create`
  takes no tags of its own.
- **A tool whose name says it makes a picture gets the space that picture will fill.**
  `makes_a_picture()` decides, generously and on purpose: guessing wrong costs nothing
  either way, because a placeholder nothing arrives for is taken down by
  `_clear_stages` when the call returns, and a generator not guessed simply appears
  without one. `_stage` draws it (a grid of dots on a `card` panel whose orange centre
  dot pulses, each pulse rippling outward, after Motion's staggered grid; sizes and
  shades are quantised to `SHADES` so the disc cache stays small); `_take_stage` removes it and says
  *where*, so `_show_preview` lands the picture exactly where its space was held.
  A preview is a `_reveal` canvas wiped in from the left, not a bare `image_create`:
  Tk has no alpha to fade, but it can uncover. The session still has to keep the
  `PhotoImage` in `preview_images` or Tk drops it when it is collected.
- **The dots between turns are `Session.thinking`.** The longest silence in a run is
  between a tool result and whatever the model does next, and it used to look most
  like nothing happening. `_begin_thinking` is idempotent and `_end_thinking` deletes
  from the mark it laid down, so the transcript is byte-for-byte what it was. They sit
  at the end, so *every* event that writes takes them down first —
  `WRITES_TO_TRANSCRIPT` is that list, checked at the top of `_handle`'s dispatch, and
  a new transcript-writing event kind belongs in it.

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
  the day it went in. **Its refusals teach**: an unknown key names the keys the
  object takes and the closest one (`compID ... did you mean compId?`), a wrong
  type or range quotes the value sent and the property's own description (which is
  where the units live), and the unknown-key check runs before the required-key
  check so a misspelling is reported as one. The reader is a 3B model with one more
  try; `is not allowed` on its own sent it guessing again, and the guess landed in
  `failed_calls`. Keep every new refusal in that shape.
- **A path from the model goes through `studio_mcp.local_path()` before the
  filesystem sees it.** A small model writes `"C:\\\\Users\\\\x"` in its arguments
  JSON for a path it was given as `C:\Users\x`; decoded, that is two backslashes and
  nothing at it, the tool errors, and the notebook learned a platitude from the error.
  `local_path` collapses runs of backslashes only when that makes something exist, so
  a path that is right is never touched; the research bridge's `resolve()` and every
  `open`/`place_file`/`import_files`/`upload_image` here call it. A new tool that
  takes a path from this PC does too.
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
python studio_agent.py --app resolve "look up the ProRes flavours and say which to deliver in"   # the sidecar in an app tab
python studio_mcp.py check --app premiere --in-process
python studio_mcp.py check --app photoshop --in-process   # against the registry entry
python studio_agent.py --mcp "npx -y some-mcp" --name Blender --list-tools  # any bridge
python studio_mcp.py check --app comfyui     # start a registry bridge, report on it
python studio_mcp.py check --app after-effects --call   # ...and call its harmless reads
python studio_mcp.py check --app resolve --snapshot tests/contracts/resolve.json
python studio_mcp.py snapshot --app resolve tests/contracts/resolve.json  # re-record
python studio_chat.py                        # the real app (console attached)
python studio_chat.py --doctor               # no window, no lock: runs beside a live copy
python studio_doctor.py                      # the same report, on its own
```

`tests/` never touches the network, the creative apps, Docker, or the model — the
research bridge's `urlopen` is swapped for a fake with a table of pages,
`test_lessons.py` drives the notebook, the reflection and the question form with the
same fake inference `test_tasks.py` uses and a temp notebook directory,
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
  tracebacks go to `studio_assistant_error.log` beside the settings file
  (`error_log_path()`: `%APPDATA%\StudioAssistant`, or wherever `STUDIO_SETTINGS`
  points), never to the screen. It sat in the source tree once, and every test run's
  deliberate `HostUnreachable` tracebacks landed beside the user's real ones.
