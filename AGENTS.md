# AGENTS.md

Working notes for anyone — human or agent — changing this project.

## What this is

A chat app that drives creative apps with a **local LLM**, one tab per app — plus
a **Chat** tab with no app behind it, for the questions that need no app: it reads
this PC's files and the web instead. Two moving parts:

- **`core/agent.py`** — the engine. The **app registry**, an MCP stdio client, an
  OpenAI-compatible LLM client (streaming and not), JSON-Schema sanitizing, and
  environment probes. Also a working CLI: `python core/agent.py --app resolve
  "what's on the timeline"`, or bare for a REPL.
- **`core/chat.py`** — the Tkinter GUI, and the way the app is actually used.
  Launched with no console via `Studio Assist.cmd` and the Desktop / Start Menu
  shortcuts.
- **`core/tasks.py`** — shared GUI/CLI execution, original-schema validation,
  bounded request context, cancellation, execution journals and task recovery.
- **`core/toolsmith.py`** — tools the model makes for itself, and the per-app
  library they are kept in.
- **`core/lessons.py`** — what the model learns per app: the `Notebook` of one-line
  lessons, the `studio_remember` tool, the reading of the user's corrections and the
  end-of-task reflection. See *What the model learns, asks and looks up*.
- **`core/mcp.py`** — the MCP harness. `Server` is the protocol every bridge written
  here runs on (framing, revision negotiation, validation, annotations, logging,
  progress, cancellation); `Loopback` is `MCPClient`'s interface over a `Server` in
  this process; `check_tools()` / `check_live()` hold any bridge — ours or installed —
  to what the executor and the inference host need. `python core/mcp.py check --app
  <id>` is the command; see *The MCP harness* below.
- **`apps/comfyui/mcp.py`** — our own MCP stdio bridge to ComfyUI's HTTP API. The
  one bridge written here rather than installed, because ComfyUI has no MCP server
  of its own and the stdlib-only rule bars the ones on PyPI. `--list-tools` prints
  its contract.
- **`apps/opencode/mcp.py`** — our MCP stdio bridge to the OpenCode server that
  `ServerSpec` starts on this PC. It follows each task to the end and puts every step
  OpenCode asks permission for to the *user*, through MCP elicitation. `--list-tools`
  prints its contract. See *The app this window serves*.
- **`apps/opencode/trainer_mcp.py`** — the trainer's MCP server, for Claude Code (not
  for the app): reads OpenCode's saved tasks, diffs and sessions, keeps and forgets
  lessons. See *OpenCode's trainer*.
- **`apps/opencode/codeaddons.py`** / **`apps/opencode/codeaddons_ui.py`** — OpenCode's Add-ons (MCP
  servers, plugins, skills): the records and catalogs, and the window. See *OpenCode's
  Add-ons*.
- **`apps/adobe/com.py`** — the road into an Adobe app that registers COM automation: a
  PowerShell worker holding `Photoshop.Application` / `Illustrator.Application`, an
  ExtendScript prelude (JSON serializer, error folding, unit pinning), and `ComHost.run()`
  which turns a script body into a decoded value or a `ComError`. See *The COM bridges*.
- **`apps/adobe/photoshop.py`**, **`apps/adobe/illustrator.py`** — our bridges to the
  Photoshop and Illustrator on this machine, each a table of tools whose bodies are
  ExtendScript run through `apps.adobe.com`. Nothing is installed inside either app.
- **`apps/adobe/cep.py`** — the road into an Adobe app that registers no COM: a CEP panel
  inside the app running a loopback HTTP server. `CepHost.run()` wraps a script body
  exactly as `apps.adobe.com` does and posts it; `install_panel()` copies a panel folder
  under `%APPDATA%\Adobe\CEP\extensions`; `explain_unreachable()` says the one
  thing to do when nothing answers. See *The CEP bridge*.
- **`apps/adobe/premiere.py`**, **`premiere_panel/`** — our bridge to Premiere Pro (the
  Beta this studio cuts in): a table of tools whose bodies are ExtendScript run through
  `apps.adobe.cep`, and the panel that evaluates them. `--install-panel` installs the panel.
- **`apps/research/mcp.py`** — the Chat tab's bridge: this PC's files and the web,
  read-only (`list_folder`, `find_files`, `read_file`, `search_web`, `fetch_page`). The
  one bridge the GUI runs *in process*, through `core.mcp.Loopback`. See *The tab
  with no app*.
- **`core/procs.py`** — every child process the app starts, contained: each in its
  own kill-on-close Windows job object, so it and everything it starts end with the tab,
  the app, or the app's crash. See *Processes: nothing outlives the app*.
- **`apps/milanote/milanote.py`** — the Milanote tab, which holds a window and has no bridge: a
  Chrome/Edge `--app` window re-parented into the tab, and uploads dropped onto the board
  over DevTools. See *The tab that holds a window*.
- **`apps/comfyui/view.py`**, **`apps/comfyui/nodes_ui.py`** — the ComfyUI tab's Nodes view:
  the same kind of window on a backend's ComfyUI, held where the transcript is, with an
  Image Studio picture's graphs loaded into it. See *The Nodes view*.
- **`apps/image_studio/imagegen.py`**, **`apps/image_studio/ui.py`**, **`comfy_workflows/`** — the
  Image Studio tab: a form (character, style, scene, references, generate) over any number
  of ComfyUI backends, with no model in the loop. See *The Image Studio*.
  **`apps/image_studio/scene/pose.py`** is its pose: OpenPose stick figures and the picture drawn from
  one, no tkinter (the editor is `PoseEditor` in the tab's module).
- **`apps/image_studio/scene/scene.py`**, **`apps/image_studio/scene/ui.py`** — the Image Studio's Scene Builder: a
  posable mannequin and simple props on a floor (and walls, each wearing a picture
  made from words), one camera, and the frame it sees, drawn as the pose and depth
  maps (and, when asked, the grey frame) the Image Studio makes the picture from. See *The Scene Builder*.
  **`comfy_nodes/studio_dwpose`** (a photo's pose points) and **`comfy_nodes/studio_facepaste`**
  (a person's real face, pasted last) are its ComfyUI nodes, kept here and copied into a
  backend's `custom_nodes`; they are not stdlib-only, they run there.
- **`apps/image_studio/scene/mannequin.py`** — the sections the Scene Builder's person is sculpted from
  (chest with pecs, waist, seat, mitten hands): functions of the angle round a bone,
  handed to `apps.image_studio.scene.scene.loft`. No tkinter.
- **`apps/image_studio/addons/civitai.py`** — LoRA profiles from CivitAI links or `.safetensors` files,
  for the Image Studio's LoRA library. No tkinter. See *The Image Studio*.
- **`apps/image_studio/addons/catalog.py`** — the Image Studio's Add-ons: LoRAs sorted by the model they
  work with, CivitAI's catalog per model, thumbnails Tk can show, uninstall to the
  Recycle Bin. No tkinter (the window is `AddonsWindow`). See *Add-ons*.
- **`apps/image_studio/addons/hub.py`** — the Image Studio's Add-ons beyond CivitAI: Hugging Face LoRAs
  for a model (adapters of its base repos, `FAMILY_REPOS`) and ComfyUI custom-node
  plugins from GitHub (topic `comfyui-nodes`), unpacked confined into
  `<ComfyUI>/custom_nodes/<repo>`, never overwriting and never running anything. No
  tkinter; the tabs are `AddonsWindow`'s "Hugging Face" and "GitHub plugins", and the
  Image Studio header's "App store" opens that window.
- **`core/icons.py`** — reads an app's own icon out of its `.exe` (PE resource
  directory → `RT_GROUP_ICON` → `RT_ICON` → DIB or PNG → resample → PNG), and
  writes the PNGs `make_icon.py` packs into the `.ico`. `struct` and `zlib` only.

### Where the work happens

Inference is **remote**; tools are **local**. That split is not negotiable:

```
this PC (the workstation)                       tailnet peer
┌────────────────────────────────────┐         ┌──────────────────────┐
│ core/chat.py / core/agent.py   │  HTTP   │ LM Studio            │
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
│   │  apps/adobe/photoshop.py ── powershell.exe ── COM ──► Photoshop
│   │  apps/adobe/illustrator.py ─ powershell.exe ── COM ──► Illustrator
│   │                                │
│   ├ MCP stdio ─┐                   │
│   │            ▼                   │
│   │  apps/adobe/premiere.py        │
│   │            │ http 127.0.0.1:7787
│   │            ▼                   │
│   │  premiere_panel (CEP) inside Premiere Pro
│   │                                │
│   ├ MCP stdio ─┐                   │         ┌──────────────────────┐
│   │            ▼                   │  HTTP   │ ComfyUI              │
│   │  apps/comfyui/mcp.py           │ ──────► │ 100.127.17.38:8188   │
│   │                                │         └──────────────────────┘
│   └ MCP stdio ─┐                   │
│                ▼                   │
│      apps/opencode/mcp.py ◄──── elicitation: the user allows or refuses each step
│                │ HTTP 127.0.0.1:4096, password
│                ▼                   │
│   ┌ opencode serve (our child) ──┐ │
│   │ works in one folder: this    │ │ ──────► LM Studio (same host as above)
│   │ repo; edits/commands ask     │ │
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
`remote`, below. OpenCode is on this machine with no window of its own: the window
starts its server as a child, and the registry calls that `served`, below.

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
`apps/research/mcp.py`'s: list and search folders on this PC, read a text document
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
  returns a `core.mcp.Loopback` over `apps.research.mcp.SERVER` instead: nothing to
  spawn, nothing to fail, no pipe to lose. `command`/`args` still name the script, so
  `python core/mcp.py check --app chat --in-process` and `--call` work on it like any
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
by `core.icons.comfy_png`, keyed by app id in `DRAWN`), and
nothing to launch. `installed()` is True for it — the tab is always worth offering —
`running()` probes its `url:`, and `launch()` raises with the `launch_note`, which for a
remote app has to say *where* to start it. The GUI asks `app.remote` before it offers
to start anything: the header button becomes **Check ComfyUI**, `_fix()` re-probes
instead of launching, and the status reads "not reachable" rather than "not running".
`tests/test_agent.py` holds a remote entry to all of that.

In the sidebar a remote app is a row like any other, but in its own group:
`detect_apps()` appends every remote registry entry with `"remote": True` and no
`exe`, and `_build_apps()` draws those last under a second heading, **ON LLM PC**
(`LLM_PC` in `core/chat.py`), after the rows for this machine. Pinning orders a row
within its group rather than across them - a pinned ComfyUI is still on the other
PC, and the heading has to stay true. The heading appears and disappears with its
rows, so hiding ComfyUI hides the group.

The URL is `COMFYUI_URL` (default `http://100.127.17.38:8188`), read once in the
engine for the probe and the bridge label, and again by `apps/comfyui/mcp.py` in its
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
- **The original's pixels are the base layer of a local edit** (`local_edit`).
  Qwen-Image-Edit redraws the whole frame at ~1 MP, and what it "leaves alone"
  comes back resized and twice through the VAE: grain, detail and colour
  drift. So unless `whole_picture` is set, the edit ends in a preview and a
  second run (no diffusion model, SAM3 at most) scales it back to the
  original's size and composites it onto the original with
  `ImageCompositeMasked` through a grown, blurred mask. The mask is SAM3's
  `region` in the picture before OR after (so "remove the man" and "add a
  hat" both work), or, with no region or none found, what changed: a blurred
  per-channel difference over `DIFF_LEVEL`. `ImageBlend`'s "difference" is a
  clamped `image1 - image2`, not an absolute one: navy made red read as no
  change in red, so both orders are screened together. Pixel difference is
  the fallback, not the plan: the edit nudges masts and roofs, so a red
  sweater came back as the sweater plus flecks of harbour. A difference mask
  covering more than `GLOBAL_SHARE` is taken as a global edit and the frame is
  kept whole, so "make it night" never comes back as night patches on day.
  Mask coverage returns to the bridge as `PreviewAny` text of a 16x16 mask (a
  torch tensor print, `mask_share`); `ImageScale` takes its size as links from
  `GetImageSize`. A photo finish on a local edit is scaled back and kept inside
  the same mask (`save_edit`). The agent prompt tells the model to name
  `region` for local edits and set `whole_picture` for global ones.
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
  to 1024, resamples it with the same model at `FACE_DENOISE` (0.3; it was 0.45)
  under `FACE_PROMPT` (the scene's prompt inside a face-specific one), with the
  sampler the next entry is about, and blends it back through a soft oval
  (`oval_png`, a greyscale PNG the bridge draws and uploads; `FACE_BLEND` sizes
  it). Why that strong and that wide: "The face pass is light", below.
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
- **A new picture's faces are redrawn with a sampler of their own; a swapped
  face is not** (a family's `redraw` in `FAMILIES`, `redraw_recipe`; 2026-09-29).
  The face pass sampled with the picture's own sampler, and Z-Image Turbo's is
  `res_multistep`, which adds no noise as it goes. The Image Studio had measured
  what that does over an existing picture ("A redraw is sampled as a redraw");
  measured again here, half of it holds and half is the reverse. On the 5090:
  two pictures of 15 faces (a wedding party of eight, faces 79-108 px high; seven
  friends at a table, 129-226 px), each saved once and its faces redrawn from
  that file with the same seeds by four samplers at 0.45 and 0.3, looked at as
  sampled (1024 px) and as they lie in the picture.
  *Skin, as found there.* `res_multistep` left dark specks on foreheads, cheeks
  and temples and white beads in beards and hair; plain `euler`, which adds no
  noise either, left the same ones in the same places; the detail recipe's
  `dpmpp_2m_sde`/`beta` left more, on rougher skin. `euler_ancestral`/`simple` at
  the same 8 steps left clean skin. In the picture the specks are a pixel or two:
  plain at x4, dirt on a face at full size.
  *Likeness, the reverse.* `euler_ancestral` does not keep the face nearest the
  one it redrew. ArcFace (antelopev2) against the face before the redraw, the
  mean of each picture: at 0.45, 0.36 and 0.53 with `res_multistep`, 0.21 and
  0.41 with `euler_ancestral`; at 0.3, 0.59 and 0.70 against 0.39 and 0.58 -
  lower on every one of the 15 faces at both strengths. `euler_ancestral` at
  0.3-0.35 strays as far as `res_multistep` at 0.45. It smooths age away with
  the specks: the wedding's guests read 12 years younger at 0.45 (3 with
  `res_multistep`), the friends, all under 35 but one, the same.
  So the two passes part. `comfy_generate`'s faces are nobody's, and its face
  pass takes the family's `redraw`: Z-Image names `euler_ancestral`/`simple`,
  the first draw and the detail pass keep theirs, a sampler the caller names is
  the picture's and not its redraws', and a family that names no `redraw` (all
  the others) redraws as it draws. The result says which
  ("face detail: ... sampler euler_ancestral/simple"). `comfy_face_swap`'s
  finishing pass keeps `res_multistep`, specks and all, because there the
  likeness is the point: on a real swap, ArcFace against the person swapped in
  was 0.75 and 0.57 before the pass, 0.47 and 0.45 after it with `res_multistep`
  and 0.41 and 0.42 with `euler_ancestral`, which also took six years off them
  (two faces only: SAM3 found the bearded man twice, and the third face was not
  a likeness before the pass either). The pass itself is what costs most: a
  third of the likeness at 0.3 with any sampler. To give the swap the same
  sampler, pass `redraw_recipe(z)` in `t_face_swap`. No sampler kept one face:
  a woman of about seventy in the first draw came back at 0.45 as a man of forty
  with `euler_ancestral` and `dpmpp_2m_sde`, and younger and mannish with the
  other two. Time is the same, 24-26 s for eight faces. Not measured on the
  3090, and on no picture with a LoRA.
- **The face pass is light, keeps the scene's words and ends inside the head**
  (`FACE_DENOISE` 0.3, `FACE_BLEND`, `redraw_faces(blend=)`; 2026-09-29). At 0.45
  the pass mended eyes and teeth and changed who the face was and what stood
  beside it. Z-Image's shift of 3 starts a denoise of 0.45 at about 0.7 of full
  noise (0.57 at 0.3; reckoned from the shift, not read off the sampler), and from
  there the model draws its own idea of a face: in a wedding party a woman of
  seventy came back a man of forty, in a family on a beach the grandmother a man
  of thirty-five. And the oval faded out only at the crop's edge, so whatever the
  redraw had made of the background went into the picture: a church where a dark
  tree was and arches in a plain wall (the prompt said "in front of an old stone
  church"), small figures in the sea behind a head on a beach, and where three
  friends sat close the largest, sharpest face of the picture blurred by its
  neighbour's redraw. Measured on the 5090 one thing at a time: seven pictures
  saved once (1824x1248 after the detail pass) and their 48 faces redrawn from
  the file with the same seeds - a wedding party (8 faces, 78-108 px high), friends
  at a table (7, 129-226), three generations on a beach (6, 90-125), colleagues at
  a window (5, 114-120), hikers far off (8, 39-57), people on a beach further off
  (8, 30-42) and a watercolour (6, 81-118) - with `euler_ancestral`/`simple`, the
  `redraw` sampler of the entry above. ArcFace (antelopev2) against the face before
  the redraw, age and sex by InsightFace, both on the face as it lies in the
  finished picture; eyes and teeth looked at, because mean |Laplacian| round them
  fell with every redraw (the first draw's grain counts as detail).
  *Strength.* 34 faces of five photographs, the whole oval: likeness 0.27 at
  0.45, 0.37 at 0.35, 0.44 at 0.3, 0.53 at 0.25 (0.66 at 0.2 and 0.77 at 0.15 on
  the 21 faces of three of them); faces ArcFace would call somebody else
  (under 0.3) 20, 10, 3, 0; years taken off 7.4, 5.8, 5.1, 4.7; sex read otherwise
  2, 0, 0, 0. Mended at 0.3 on every face, the 30-42 px ones too, which were
  broken before the pass, eyes garbled and glasses half there. At 0.25 faces of
  78 px and more are mended as well and the 39-57 px ones nearly, but five of the
  eight smallest kept smudged eyes or the ghost of their glasses; at 0.2 eyes go
  soft on the larger faces too. So 0.3: no harder, and no lighter.
  *Words.* Taking the scene out of the face prompt was the obvious cure for the
  church, and it cured it - and cost likeness on every picture (0.24 against 0.27
  at 0.45, 34 faces) and age wherever the prompt says how old people are: the
  family lost 9.7 years without it and 4.2 with, and at 0.3 the grandmother was a
  woman with the scene and a man of twenty-seven without. Only the scene's light,
  only the realism sentence, and a prompt without `FACE_PROMPT`'s list of what a
  face has all came out as none did (wedding, 0.3-0.45). The crop carries the
  light; the words carry who the people are. They stay.
  *Oval.* `FACE_BLEND` is `oval_png`'s scale and centre, 0.75 at 0.555: full
  strength over the face, nothing beyond 0.35 of the crop from the face's
  middle. Mean change outside every head (0-255), 21 faces at 0.3: 0.77 through
  the whole oval, 0.12 at 0.85, 0.02 at 0.75, 0 at 0.65 - where the oval cuts into
  the face itself (the picture's face 1.27 from its own redraw inside the box,
  against 0.6-0.8). At 0.45 the church still showed through 0.85 and as a ghost
  through 0.75; at 0.3 through 0.75 the wall beside a head is the first draw's,
  stone for stone, and the friend at the table is as sharp as she was drawn. No
  seam in hair at any size. The face swap's pass keeps the whole oval (it blends
  a pasted head, hair and all, and was not measured); each size is a file of its
  own on the server (`studio_face_oval_75_555.png`).
  *Together*, 42 faces of six photographs, as it was against as it is: likeness
  0.24 and 0.44, faces under 0.3 28 and 4, ten years younger or more 15 and 9,
  sex read otherwise 2 and 0, the face's colour moved 5.1 and 3.2 (Lab), change
  outside heads 0.86 and 0.01. With `res_multistep`, which sampled this pass
  before the entry above, the same settings keep more and leave its specks: 0.62
  against 0.44 and 2.4 years against 5.1 (23 faces). `comfy_generate` run for
  real with both changes in (the wedding's prompt and seed, 49 s, 8 faces at 0.3
  with `euler_ancestral`/`simple`) gave the experiment's picture pixel for pixel.
  Seen and not fixed. The pass still takes six years off a face and ten or more
  off one in five; the woman of seventy is at 0.3 somebody of fifty who could be
  either. A strength by size would keep more - 0.25 mends everything from 78 px
  up, at 0.56 against 0.47 - but where between 57 and 78 px to change over was
  not measured. A painted picture loses its paint: the watercolour's faces came
  back smooth at every setting, scene words or none, and through the smaller
  oval the smooth face now sits in painted surroundings. Faces of 130 px and
  more were good before the pass, which costs them likeness and gains little.
  Where heads touch, a neighbour's oval still reaches the face. Not run: the
  3090, a LoRA, the chat tab's own model calling the tool.
  Pictures, sheets and scripts: `D:\ComfyUI\output\ImageStudio\_exp\chat_identity`.
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

The **Image library…** button (`ImageLibraryWindow`) is every past generation,
newest first, built fresh from `self.studio.history.list()` each time it opens or
its search box changes - not a separate saved collection (`Library.all("images")`,
`images.json`, still exists for `Library.import_image`/`register_image` and their
own tests, but no UI reads it any more). One entry per output picture; a batch of
several from one job gets "(2/3)" appended to its prompt so they stay distinguishable.
Search matches the prompt, not a filename. Picking one assigns it to an existing
reference slot (source, style, pose, …), so normal workflow support checks still
apply.

The ComfyUI tab is a conversation; the Image Studio (`IMAGE_STUDIO`, an `ImagesSpec`,
a `PanelSpec` with `images = True`) is a form. It holds no other program's window,
so `_ensure` just marks it ready and calls `ImageStudio.start()`, the first thing in
it that touches the network (a tab is built before it is looked at, and the GUI
tests build every tab). `apps/image_studio/imagegen.py` is the engine, with no tkinter;
`apps/image_studio/ui.py` is the tab, a collaborator the window lends `_skin`, `_button`,
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
  the WebSocket (`apps.milanote.milanote.WebSocket`, read on a thread of its own so a timeout
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
  `python apps/image_studio/imagegen.py --probe` checks `/system_stats`, `/object_info`,
  `/prompt` (an empty graph, which ComfyUI refuses without running anything),
  `/history` and the WebSocket on every backend, then prints each model's
  readiness.
- **Workflows are files.** `comfy_workflows/<id>.json` is an API-format graph plus:
  `{{placeholders}}` (a whole value keeps its type), `_when`/`_unless` nodes (a
  `_when` list keeps the node when any of them is set), `switches` (a link chosen
  by a value; a branch may be `"{{other}}"`, a switch named before it), `lora_chain` (where LoRAs hang;
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
  LoRA of unknown family is applied with one. Nothing is dropped silently — except an
  **"Always on"** LoRA (`always` in the library): it joins every picture whose model it
  suits, at its library strength, and is skipped quietly for other families, because
  "suits" is the rule the user set. One added by hand keeps the form's strength.
  Since Add-ons (2026-09-26) the *form* no longer offers an incompatible LoRA at all,
  so these warnings are for what reaches `compose` another way: an identity's or a
  style's LoRA, or a history record's settings.
- **A LoRA mix is a preset.** "Save as preset…" under the form's LoRAs stores the
  rows and strengths (`presets.json`, `clean_preset`) on the built-in preset then
  chosen (`base`: refine, face pass, size, routing role). It is listed in the Preset
  row after the built-ins; picking it loads its LoRAs *into the rows*, replacing the
  last mix's rows (hand-added rows stay), and a built-in takes them away. The rows,
  not the preset, are what `compose` uses, so the strengths can be nudged before
  Generate. Everything that looked a preset up goes through `preset_info(lib, key)`;
  an unknown key is Standard, and a mix can never take a built-in's key (`mix-`).
  An added row's whole `trigger` goes into the prompt, so a CivitAI "trained words"
  list of alternatives (an expressions LoRA's) must be cleared, not kept.
- **A LoRA stack is held to what the model takes** (`hold_loras`, a workflow's
  `lora_budget`; 2026-09-29). Stacked LoRAs add up, and a step-distilled model
  has little room: on Z-Image Turbo four Always-on LoRAs at the 0.8 each was
  imported with (3.2 in all - the library as it was that day) made "a red fox
  in fresh snow at dawn" a night scene with a grid across it, a grey knit
  sweater a bra under a cardigan, and a Scene Builder picture (pose and depth
  maps) a smear with no people in it. Each LoRA alone at 0.8 was fine, and so
  were any two. The same four scaled to a total of 2.0 gave a picture again,
  dark and oddly dressed; at 1.5 a dim one; at 1.2 a clean one that still
  had their look; below 1.0 they did little. So `zimage_hq` names
  `lora_budget` 1.2, and FLUX names none (nothing was measured there). Over
  the budget **only the Always-on ones are turned down**, all by the same
  share, into what the others leave: nobody chose their sum - each was
  switched on alone in Add-ons. A LoRA chosen for the picture (a row of the
  form, an identity's, a style's) keeps its strength; a sum of those over
  the budget is said and not changed, and leaves the Always-on ones out. A
  warning names each and its new strength, and that fewer Always on leaves
  each stronger. `settings["lora_budget"]` is a picture's own (0: no
  limit); History keeps the strengths used, so Generate Again is the same.
  Community advice agrees on the cause and differs on the number ("under
  1.0" to a normalised 0.7-0.9); 1.2 is what was measured here.
- **A trigger is said for a LoRA the model is given.** `compose` used to add
  the trigger of every LoRA in the stack, the ones it then left out too: an
  Always-on LoRA whose file was not on the 5090 was left out with a warning
  on every picture while its trained words ("ultra detailed, cinematic, ...
  detailed skin pore, ...") went into every prompt, a fox's included.
- **LoRAs come in from CivitAI** (`apps/image_studio/addons/civitai.py`, the LoRA library's *Import
  from CivitAI…*). Paste links (a model page, `modelVersionId`, a download link, an
  AIR, a bare version id) and/or pick `.safetensors` files. A link is read from
  CivitAI's public API: primary file, trained words as the trigger, `baseModel` as
  the family (`BASE_FAMILIES`), tags as the category, the description as notes, and
  the tamest still image as the preview (`image-studio/lora-previews/`). A file is
  read for its own header (`ss_*`, `modelspec.*`), hashed, and looked up by SHA-256
  (`/model-versions/by-hash/`); offline or unknown, the header is the profile.
  `Library.import_lora` matches by hash, then filename, and fills only *empty*
  fields of an existing record, so re-importing never overwrites what the user
  wrote. The file itself goes only into a backend's `lora_dir` that is on this PC
  (a download for a link, streamed to `.part` and checked against CivitAI's hash;
  a copy for a file), or nowhere. The API key (many downloads need one) is
  `CIVITAI_API_KEY`, else `image-studio/civitai.json`. Tests: `test_civitai.py`,
  against a table of canned answers.
- **Identity and style are separate records.** An identity is a LoRA, a trigger, a
  strength and reference photos (copied under `image-studio/references/`). A style
  is a LoRA and/or prompt additions plus look defaults. The precedence is model
  defaults < style < preset < the form. `compose()` builds the prompt as the person
  (identity triggers, then the look, `person_text`: a bare "auburn" becomes "auburn
  hair"), then the scene, then the camera line, then the anatomy constants, then the
  style. It is pure and does no I/O, which is how the form
  shows warnings before Generate.
- **What the pipeline adds to the user's words is drawn, like any other
  words** (2026-09-29, measured on the 5090, the same seeds, only the added
  words changed). Every shipped workflow samples at CFG 1: there is no
  negative prompt, and a thing named to rule it out is a thing named. So what
  `compose` adds is short, says what is there, and goes on the person it is
  about - see the clothing floor and the anatomy constants below. **Nothing
  is added that nobody asked for.** On 2026-09-28 a no-style prompt was to
  get a light ("golden hour, warm low sun") and a human-error detail ("a
  loose thread on a sleeve") picked by the seed (`time_of_day_text`,
  `imperfection_text`). It never ran: it was skipped "whenever a style is
  chosen", and "No style" is itself a style record, the form's default. Run
  live before switching it on, it made the flaw the subject - the head cut
  off to show a sleeve, 6 of 12 pictures - and put a lamp post in a park and
  a pendant lamp over a kitchen for "a single lamp at night". Both functions
  are gone. A richer prompt is the user's to write, or a style's.
- **Pick person cuts one person out of a reference photo.** In the Identities editor,
  the photo is the selected reference, else one chosen from disk. `Studio.look_at`
  sends it through ComfyUI for its size and a PNG (Tk reads no JPEG; no SAM3), and a
  crop window (`_crop_photo`) takes a drag round the person (`crop_region`; a slip
  under `CROP_MIN` px is ignored) or "Whole photo". `Studio.find_people(path, region)`
  then runs SAM3 (`person:8`) on that crop only, on the first online backend with a
  sam3 checkpoint; with more than one person a window shows numbered boxes and a
  click picks (`pick_box`). `cut_person` crops round that box (offset back into the
  photo), masks the person with SAM3 **aimed at the picked box** (`bboxes`) and lays
  them on white, so a neighbour's shoulder goes too. Without the box, "person:1" in a
  tight crop once masked a neighbour's hand at the edge instead of the woman filling
  it. The cut-out takes the photo's place in the list (the front for a photo from
  disk; first = face reference); a listed photo stays after it. The runs are polled
  on `/history`, not the job queue: about a second each.
- **The look is a video game's character creator.** `LOOKS` is the sections (Body,
  Face, Hair, Expression, Clothes, Accessories) of slots, each `(setting, label,
  nouns, picks, many)`; every slot also takes free text, and a `many` slot
  (accessories, marks) toggles picks in a comma list (`toggle`). Weight, muscle and
  height are `SLIDERS`, -3..3, nothing said at 0. Height's setting is `stature`:
  `height` is the picture's. Expressions show as emoji (`EMOJI`), on the form as
  faces alone; the emoji never reach the prompt. A **character** (`characters.json`,
  made in `CharacterCreator`) keeps every slot and slider but `PER_PICTURE`
  (expression, gaze), an identity for the face, and a picture per item worn
  (`item_refs`, copied under `references/`). Choosing one copies its look onto the
  form (blanking what it does not set) rather than linking to it, so history holds
  the whole look and Generate Again does not change when the character is edited.
  The People tab (2026-09-26) has **one** person dropdown: characters, then
  profiles (identities) alone, `"c:<id>"` / `"i:<id>"` (`_pick_from_people`); a
  character brings its profile or none, and picking another profile clears the
  character. Under it, **Editor** (the creator) and **Image references** (the
  profiles). The form's look tabs are `FORM_LOOKS`: Body and Accessories are the
  creator's alone, though a character's body and accessories still reach the prompt.
- **A character's tags are words for its pictures** (2026-09-27, the user: "tags need
  images to upload. so when i say glasses i reference the image"). The creator's
  Tags tab is for things on the person (the user: "glasses, earrings, dress, tattoos,
  etc."). It lists every `item_refs` entry (the item pictures are tags too) and makes
  one from a word plus a picture uploaded then: no picture, no tag. The copy goes
  under `references/` (`keep_reference`), and *From library* gives this character a
  tag another character already has, sharing its copy. `outfit_of` counts a tag
  when a slot holds it exactly (as before) or when `says` finds its word, any case
  and a plural allowed, in a `TAG_SLOTS` slot (the item slots and Traits, where
  tattoos are) or the scene ("round glasses", "she pushes her
  glasses up"; not "sunglasses"). The longer tag goes first and uses up
  its words, so "a rose tattoo" brings the rose tattoo's picture and not a plain
  "tattoo" tag's as well. A tag said only in the scene is clothing when its
  word names a Clothes pick's garment and nothing on the head, else an accessory.
  The pictures then go through Kontext as any item picture does. Scenes with people
  still blank `item_refs` (`apps.image_studio.scene.scene.generation`): tags are the form's person's.
- **Item and person pictures can come from a link** (2026-09-27, the user: "i want to
  use urls for images of items and people"). Beside every Picture… on an item row
  (the form's and the creator's, Tags included) is **Link…**, the Tags tab has
  **From link…**, and a profile's reference photos and profile picture have **Add
  from link…** / **Link…** (identities only; a style tile or LoRA preview does not).
  `ImageStudio.from_link` asks in `_ask_name`'s window, pre-filled from the
  clipboard when it holds one link, and downloads on a worker thread
  (`Library.keep_link`), posting the path back only while the window is open, and
  only to the record it was asked for. The picture is then a file under
  `references/`, named by its bytes as an upload is (`keep_bytes`), so the same
  picture by link and by file is one file and a link that dies later breaks
  nothing; history never holds a URL. `fetch_picture` takes the picture's own
  link, a page that names one (`og:image`, `twitter:image`: a shop's or a
  profile's page is what gets copied), a Google Images result (`imgurl`) or a
  `data:` link, with a browser's user agent (`PICTURE_AGENT`, for the reason the
  research bridge has one). What came must be a picture by its own first bytes
  (`picture_ext`: PNG, JPEG, WebP, GIF, BMP), never by the server's Content-Type;
  AVIF/HEIC, a page with no picture, a 403 and anything over `PICTURE_BYTES` are a
  `LinkError` that says what to do instead. Tk shows only PNG and GIF, so a JPEG
  from a link has no thumbnail, as a JPEG from disk has none.
- **Item pictures go into the picture itself, through FLUX Kontext**
  (2026-09-25; before, Try On redrew the finished picture, which the user did not
  want). They are chosen on the form's Clothes, Hair and Accessories tabs as in
  the creator (`item_rows`; the Hair tab keeps one picture, item name `hair`),
  and copied under `references/`. `plan_items` takes the pictures of what is
  worn today (`outfit_of`: clothes, hair, accessories; anything else skipped
  without a word). When the workflow has an `items` section (the FLUX
  baseline) and the backend has `kontext_model`
  (`flux1-dev-kontext_fp8_scaled.safetensors`, Comfy-Org's fp8: BFL's full
  weights are gated), the model file becomes Kontext, guidance
  `item_guidance` 2.5 unless the form sets one, and `add_item_refs` puts the
  pictures side by side on white (`ImageStitch`), through
  `FluxKontextImageScale` and `VAEEncode`, onto the prompt as one
  `ReferenceLatent`. One picture of them all, since Kontext [dev] was trained on
  one reference. The prompt gains `ITEM_PROMPT` ("look exactly as in the
  reference picture ... one person, not the reference picture itself"), which
  the face pass's prompt leaves out. The LoRA chain, pose ControlNet, refine and
  face pass all run on Kontext unchanged. Without Kontext, or with a workflow
  that has no `items` section, a warning names the missing piece and the
  words alone describe the items. Measured on the 5090 (832x1216): 17-18 s, 21
  s with a drawn pose (which it follows). A dirndl photographed on a model came
  out right in every detail (lacing, apron, trim, lace hem) across scenes and
  poses. But her pendant came along, and without an identity LoRA the face
  drifted toward hers. Pictures of the item alone, on white, are best.
- **A character is a profile: its face photos draw every picture of it**
  (2026-09-26, the user: "profiles for people to face swap"). **The photos are
  managed in the identity builder only** (Image references: "Reference
  photos — first photo is Primary"). The user, 2026-09-27: "those controls should
  only show in the identity builder. the drop down is the correct format". A
  first version put a Face photos row in the Character creator and a strip
  with a "Their face" tick on the form. Both were removed when the branch was
  ported onto the People section's single person dropdown. The first photo is
  PuLID's, and the real-face paste chooses among all of them. `character_faces`
  reads a character's legacy `faces` if it has any, else its identity's
  reference photos. The dropdown copies them onto the form without showing
  anything (`face_photos`, `face_name`): a character ("c:") gives its
  identity's photos, a profile alone ("i:") its own, and no one clears them.
  History holds them, so Generate Again redraws the same face. The WithAnyone
  recipe draws identities together and turns the face pass off; it wins over
  the photos. `faces_of` turns them into the Scene Builder's `scene_faces`
  shape - one person, `at` None (the biggest face the finder finds), the
  whole frame as their region, `FORM_LIKENESS` 0.6, real faces on - and
  forces the face pass; a scene's own `scene_faces` win, and a scene blanks
  the form's. In the Scene Builder a person's own face picture comes first,
  then their character's photos, then its identity's. **A whole-frame
  PuLID gets no attention mask**: with Kontext's item picture the latent
  has the reference's tokens too, and a mask the picture's size failed
  ("tensor a (8022) must match ... (3952)"). Measured on the 5090, 832x1216,
  Partner's 5 head crops: 42 s alone, 59 s with a dirndl picture through
  Kontext, whose face no longer drifts to the garment model's. `thumbnails`
  (PNGs made by System.Drawing, one PowerShell run; Tk reads no JPEG) stays in
  studio_imagegen for any photo strip. Nothing on the form uses it now.
- **The anatomy constants** (`ANATOMY`): a picture with a person in it (chosen,
  described, or named in the scene, `PEOPLE`) says that every person has
  two hands, each with four fingers and a thumb, two feet, two eyes and a
  proportionate body. It is a positive sentence on purpose: every shipped
  workflow samples at CFG 1, which ignores the negative prompt, so the matching
  negatives are added only for a model that reads one. On by default; the form
  has a switch (`anatomy`). Two things were found live on 2026-09-29. The
  sentence began "Drawn correctly:", and that word made photographs
  illustrations - 2 of 6 on FLUX.1 [dev], 3 of 24 on Z-Image Turbo, none
  without it - so it is gone. And **a workflow can leave the sentence unsaid**
  (`"anatomy": false`, `zimage_hq`): Z-Image Turbo, told of hands and fingers,
  made them the picture - a gardener and a knitter cropped to their hands, the
  head out of frame, 6 of 9; a runner became an open hand held up to the
  camera - and untold it drew the same hands well. A job note says so when
  the switch is on and the workflow leaves it out. FLUX is still told: there
  the sentence changed nothing else in the picture.
- **Nobody is undressed** (`COVERED`, `CLOTHED`, `is_dressed`): a picture with a
  person described (the form's person, an identity) says `CLOTHED`, "fully
  clothed", after them, and "wearing clothes suited to the scene" too when
  the form's Clothes (top, bottom, outerwear) and the scene name no garment
  (`GARMENTS`). A scene that describes its people itself has no one to hang
  them on: it gets "Fully clothed." after it, a sentence of its own.
  "Natural anatomy" and the chest words alone drew people nude; "plain
  underwear" was tried first and put people in underwear at Oktoberfest. It
  is a floor, not a switch: the anatomy switch does not turn it off, and a
  scene asking for bare skin still gets it. **The floor names no skin and no
  swimwear.** Until 2026-09-29 `CLOTHED` was "Every person is clothed, the
  chest fully covered by their clothing or swimwear", added because "partner in
  a swimsuit" after "very full chest" was drawn topless. It drew what it
  named: on the same seeds a man knitting in an armchair sat shirtless in
  briefs, a runner and a gardener wore swimsuits (9 of 9), and the form's
  person walking a dog in a park, no garment named, was topless (3 of 3).
  With "fully clothed" on the person all of them were dressed, and a red
  swimsuit that was asked for was still a swimsuit, on 3 of 3. And
  `COVERED` used to open the prompt by itself when nobody was described
  ("wearing clothes suited to the scene. An elderly man..."), a wearer-less
  clause where the model weighs words most. The anatomy constants no longer
  say "anatomically" or "natural anatomy" for the same reason.
- **Styles are chosen by picture.** The form shows each style as a tile of one cat
  photo in that style (`style_example`): the style's own `example` (a PNG), else
  `style_examples/<id>.png`, else a blank tile with its name. The shipped ones
  are 208 px squares made on the 5090 by FLUX image to image from that photo,
  with the style's prompt first (after the subject, it barely registered). Denoise
  was 0.5 for Modern photograph and ~0.93 for the rest; below that, image to image
  keeps the photo's colour and "Black and white" comes out in colour. A new
  default style needs its tile (a test checks). Tk scales pictures only by
  whole factors, so `photo_at` zooms and subsamples to hit the tile size, and
  passes a master: a `PhotoImage` without one belongs to the first Tk root.
- **A reference is typed** (face, pose, composition, style, source) and used only
  where the workflow declares that kind. Otherwise it is a warning, not a silent
  reuse. As of 2026-09-25 `flux_hq` takes `source` (image to image) and
  `style`/`composition` (Redux, when its two files are on the backend), and the
  FLUX baseline takes `pose` (below). Nothing does face conditioning yet (no PuLID
  or IP-Adapter in a shipped workflow), so the likeness is the identity LoRA's.
- **The camera is aimed on a diagram, not typed** (`CameraAim`, the Camera row).
  From above, the camera is dragged round the person (`turn`, 45° steps toward
  their left); from the side, up and down (`height`) and in and out (`shot`,
  face to wide). It is `settings["view"]`, None until touched, so an untouched
  form prompts as before. `view_text` goes *first* in the prompt, since FLUX
  weighs the start most, and every shot but the face says the whole head is in
  the frame with space above it: asked for because FLUX, left alone, cut heads
  off. With a drawn pose only the height is said (and a warning says why): the
  figure already frames and faces the person, and words that disagree fight the
  ControlNet. The free-text Camera field stays, for lens, light and film.
- **The pose is a stick figure the user drags** (`PoseEditor`, the Pose row's
  Draw…). It is OpenPose's 18 body joints in its colours on black, because that
  is the picture pose ControlNets were trained on; `apps.image_studio.scene.pose.render` draws it
  the way OpenPose's own preprocessor does (limbs at 60%, joints full), with
  `struct` and `zlib`, in ~30 ms. Dragging a joint carries what hangs off it
  (`CHILDREN`), Shift moves it alone, the empty frame moves the figure, the wheel
  resizes it, right-click hides a joint (it stays placed, so it can come back;
  hidden limbs are not drawn). Points are fractions of the frame, and the frame
  is the picture's size as last composed (`planned_size`); a size changed since
  the pose was drawn is `refit` at Generate - scaled and centred, never
  stretched - and redrawn. The picture is named by a hash of the points and size
  under `image-studio/poses/`, and the settings keep the points, hidden joints,
  size and strength (`pose`), so Reuse Settings reopens the figure, not a PNG.
  A picture chosen with Choose… replaces the drawn pose; it must already be a
  skeleton, since no preprocessor (DWPose) is installed.
  `flux_dev_baseline` applies it with `ControlNetApplyAdvanced` (nodes 50-52;
  the loader, 50, is shared with the depth map's 53-54) to the first pass only, at `pose_strength` 0.9 to
  `pose_end` 0.65 of the steps - Shakker's recommendation for pose on Union
  Pro 2.0, which leaves the last steps to the model's own detail. The refine
  and face passes keep the plain conditioning. The ControlNet file is named in
  the workflow's defaults, not the model record (so a `models.json` saved
  before it still finds it: compose `borrowed`), and is kept in the values -
  and the record - only when a pose is used. Without the file the pose is a
  warning and the picture is made without it; with no pose the graph is the
  baseline node for node.
- **A pose's face needs its 68 dots, or the person turns away.** Measured on the
  5090 (2026-09-25, seeds 7, 99 and 4242): a body-only skeleton came back seen
  from behind every time - swapping left and right did not change it, and
  neither did spreading the eyes. Union Pro 2.0 learnt its poses from DWPose,
  which draws a face's 68 landmarks whenever it sees one, so a head without them
  reads as the back of one. `face_points` puts a generic frontal face (`FACE`,
  iBUG order) on any figure whose nose and both eyes are shown, turned and sized
  by the eyes; with it all three faced the camera. The head must stay a real
  one's size (eyes ~1/27 of the height apart): doubled, the dots gave caricature
  heads. The profile preset, with one eye, gets no face. The picture's name
  includes `DRAWING`: bump it whenever `render()` changes, or a pose drawn before
  keeps its old cached picture - which is how the first face-dot test ran
  against the old drawing. A posed FLUX picture at 832x1216 took 14-15 s on the
  5090 against 12 s without.
- **The editor draws a mannequin; the model gets a skeleton.** Since 2026-09-25
  `PoseEditor` draws a wooden artist's mannequin (tapered limbs on ball joints,
  a torso, a head that shows its facing, hands with fingers; the person's right
  darker) from the same 18 points, and "What the model sees" shows the skeleton
  `render` makes. Hands are DWPose's 21 points (`hand_points`), carried on from
  the forearm at `HAND_LENGTH` of it, palm to the viewer unless `back`. A shape
  (`HAND_SHAPES`: relaxed, open, fist, grab, point, peace, thumbs up, OK) is
  bends per finger joint, foreshortened as a bend toward the viewer would be,
  so a fist's tips come back onto the palm. A click on a hand (not a drag)
  gives it the next shape; the menus beside the frame do too, and turn it over.
  `pose["hands"]` holds each side's shape and `back`; a pose saved before hands
  existed has none and keeps its picture's name. What held on the 5090 (seed
  4242 and others, a full-length man, hands ~70 px): the skeleton alone gave
  open, pointing and a fist on an outstretched arm, but not peace, thumbs up or
  a fist on a hanging arm; holding the ControlNet to 85% of the steps changed
  nothing, and thinner hand lines lost the fists (the dots are now ~0.034 of the
  hand's span). So compose also writes the shapes into the prompt (`HAND_WORDS`,
  `hands_text`): one shaped hand then comes out right (peace, thumbs up), but
  two different shapes bleed - the stronger gesture lands on both hands - and
  FLUX does not keep the person's right and left apart in words. The next step,
  if hands must be exact, is a hand pass like the face pass: the hands' places
  are known from the pose, so each can be cropped, redrawn at 1024 px with its
  own skeleton crop and its own one-hand prompt, and blended back.
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
- **FLUX runs the baseline, with layers that leave it alone when off**
  (`flux_dev_baseline.json`): ComfyUI's own `flux_dev_full_text_to_image`
  recipe plus FluxGuidance. Layered on it, one at a time (2026-09-25): a
  `lora_chain` (identity, style and added LoRAs, `LoraLoader` on model and
  clip), a `refine` pass (flux_hq's upscale and tiled low-denoise redraw), and
  the face pass (below). With no LoRA, refine or face pass the filled graph is
  the baseline node for node; a test holds it to that. Image to image (a
  `source` reference, nodes 21-22, at `denoise`) and a `composition` reference
  - a depth map through the pose's ControlNet, chained after it (53-54, at
  `composition_strength` 0.55 to `composition_end` 0.5) - came with the Scene
  Builder's maps (2026-09-25). Redux is still only in `flux_hq.json`, which the
  tests keep covered through a `flux-hq` model of their own. A denoise below 1
  with no source picture is put back to 1 by compose and said: over an empty
  latent it only leaves noise in the picture.
- **The face pass** is the chat bridge's face detail (`apps.comfyui.mcp.
  face_detail`) for a template with a `face_detail` section. The run that
  makes the picture also runs SAM3 on it (`add_face_finder`, nodes `fd*`);
  a second run (`face_graph`) loads the saved picture, crops each face
  `FACE_PAD` times its size, redraws it at `FACE_EDIT` px with the job's own
  model, LoRA chain and guidance at `face_denoise` (0.4 for FLUX), and blends
  it back through a soft oval. The identity LoRA is on the model that redraws
  the face, which is where the likeness sharpens. `face_graph` fills the
  template with the section's nodes and keeps only the nodes the redraw links
  need, so the picture is not made twice. It is a finish, never a reason to
  lose the picture: no SAM3 checkpoint (`sam3` in its name, under
  `checkpoints`) or a missing node turns it off with a warning, and a failed
  second run keeps the first run's picture with the error in the record.
  Identity Portrait and High Quality Final turn it on; the form has a toggle.
  Faces get seeds s+1, s+2, ... so Generate Again redraws them the same.
- **A redraw is sampled as a redraw** (a workflow's `redraw_sampler`,
  `redraw_scheduler`, `redraw_steps`; `face_graph`'s KSampler; 2026-09-29).
  Every part redrawn over what is there - the face pass, the hands, the eyes,
  the glasses, Fix a spot, the Critic's fixes - used the picture's own
  sampler. Z-Image Turbo's is `res_multistep`, which adds no noise as it
  goes, and over an existing picture it leaves dark specks on skin and white
  beads in beards and hair at every strength from 0.3 to 0.6, with a plain
  prompt too (`euler` and `gradient_estimation` the same); the detail
  recipe's `dpmpp_2m_sde`/`beta` left dry, scaly skin and at 0.6 changed
  who the face was. The ancestral samplers leave skin. **But a redraw that
  leaves skin drifts further from the face it redrew**: measured by ArcFace
  on ten faces of 70-250 px, twelve samplers at strength 0.4, likeness to
  the face before was 0.71 for `res_multistep`/beta, 0.70 `euler`, 0.69
  `res_multistep`/simple, 0.67 `euler_ancestral`/beta, 0.64
  `euler_ancestral`/simple, 0.54 `lcm` - the samplers that speckle keep the
  geometry, the clean ones move it (and ArcFace does not see specks). A
  first write-up here said `euler_ancestral` "kept the face nearest"; that
  was by eye, and the chat bridge's own measurement
  ([[chat-bridge-redraw-sampler]] in the memory notes) found the reverse
  first. So the redraw is `euler_ancestral` on the **beta** scheduler, which
  was nearer than simple on 6 pictures of 6 and shifts the age read off a
  face by -0.1 years against -0.9, and **the Z-Image face pass runs at
  `face_denoise` 0.3**, not 0.4: likeness 0.775 at 0.3 against 0.670 at 0.4,
  nearest on 7 faces of 10 - nearer than `res_multistep` at 0.35 - and it
  still mends the eyes and teeth of 60 px faces. Hands at 0.4 on beta were
  as good as on simple on 7 pictures (a ring kept, small hands sharpened).
  The eye pass after the face swap costs the swap's likeness 0.015-0.018
  with `res_multistep` and 0.03-0.04 with `euler_ancestral` (three
  pictures); it takes the redraw sampler all the same, for the specks under
  the eye, and the number is here for whoever wants it back. `zimage_hq`
  names all three and keeps `res_multistep` for the picture (ComfyUI's own
  recipe, unchanged). FLUX names none and redraws with its own `euler` at
  0.4, as before (not measured there). A sampler typed in Advanced is the
  picture's, not its redraws'.
- **The face pass, like the hands pass, is for pictures whose words name a
  person** (2026-09-30). Its prompt draws "a real human face ... natural lips
  and teeth" on whatever SAM3 calls a face, and a fox's is one to it: High
  Quality Final on "a red fox in snow" gave the fox a person's mouth.
  `compose` turns the pass off, with a note, when neither the form, an
  identity, a scene's people nor the scene's words (`has_person`) name
  anyone; the rest of the preset (the refine pass) stays.
- **The Z-Image refine pass enlarges with an upscale model** (a workflow's
  `refine_model`; `zimage_hq` nodes 46-49; 2026-09-30, the user: "download the
  upscale model and finish the refine pass"). The pass was ComfyUI's own
  Z-Image upscaler recipe less its model: lanczos, then a redraw that had a
  blur to sharpen. `RealESRGAN_x4plus.safetensors` (Comfy-Org's repackaging,
  BSD-3, 66,857,836 bytes, sha256 37f9a931...9a60bb checked against Hugging
  Face) is in `D:\ComfyUI-models\upscale_models` on the 5090; ComfyUI listed
  it without a restart. The picture goes through it (x4), comes down to the
  size asked (`refine_model_by` = upscale / 4), and is redrawn. Three things
  were measured rather than taken from the template (five pictures, then
  three portraits through the engine with the library's LoRAs; ArcFace
  against the plain enlargement, and the skin's fine texture):
  - *The redraw is `euler_ancestral`/beta, 5 steps, at 0.2*, not the
    template's `dpmpp_2m_sde`/beta at 0.33, which left scales and specks on
    skin and fur and kept 0.59 of a face's likeness (the same with lanczos
    or the model under it). At 0.2 it keeps 0.82-0.89 and as much fine
    texture as the base picture had. Lower is nearer still (0.93-0.96 at 0.1)
    but leaves the model's scratch-like hairs on knitwear; 0.25 and 0.3
    smooth the skin more, not less, and drift further. So 0.2 is the least
    that takes the model's artifacts out.
  - *The model's enlargement is laid half and half with lanczos's*
    (`refine_blend` 0.5, `ImageBlend`). Alone it smooths skin and oversharpens
    fur; half and half was nearest on likeness (0.84 against 0.83 alone and
    0.78 for lanczos at the strength that needs) and kept faint freckles the
    model alone lost. A small gain, one node.
  - *Without the model on a backend* (`inventory`, or the two nodes) the
    pass is lanczos as before, redrawn at 0.25 (`refine_model.without`), and
    a warning names the file and its folder. It is never a reason to fail,
    and never part of a model's readiness. The 3090 has no upscale model.
  The record's `refine` says what ran (`model`, `sampler`, `scheduler`,
  `steps`, `denoise`), and History's line names the model. 2x on a 1024²
  picture adds about 10 s on the 5090. FLUX's refine pass is unchanged:
  tried on three FLUX pictures, the model made little difference there (its
  pictures are soft-focus by design), and 0.2 against its 0.3 kept faces
  nearer (0.95 and 0.90 against 0.89 and 0.84) - too few pictures to change
  it on.
- **The 5090's ComfyUI (2026-09-25)** is ComfyUI v0.37.2 (the 3090's version),
  git-cloned into `D:\ComfyUI` with its own Python 3.12 venv and PyTorch
  2.11.0+cu130 (Blackwell; cu128 until 2026-09-29). Keep torch on the CUDA
  major that onnxruntime-gpu is built for (1.30: CUDA 13): with cu128 torch its
  CUDA provider could not load `cublasLt64_13.dll` and every InsightFace
  session asked for CUDA silently ran on the CPU (PuLID's face analysis ~2 s a
  photo, 35 ms on CUDA). The pre-switch `pip freeze` is
  `D:\ComfyUI\venv-freeze-cu128-2026-09-29.txt`. The models live in `D:\ComfyUI-models`
  (`extra_model_paths.yaml`), so a reinstall keeps them. It is started by
  `D:\ComfyUI\Start ComfyUI (Image Studio).cmd` on 127.0.0.1:8188, and the app
  does not start it. FLUX files: `flux1-dev.safetensors` (Comfy-Org mirror, the
  same file as BFL's), `clip_l`, `t5xxl_fp16`, `ae`. Measured: 1024², 20 steps,
  ~2.6 it/s, 11-12 s end to end including the model load. The 3090 has no FLUX
  files (its entry expects `t5xxl_fp8_e4m3fn_scaled` and fp8 weights), so Auto
  never sends FLUX there. Z-Image Turbo at 1024² took 18-30 s there.

### Identity profiles and FaceFusion

People's **Choose head photo…** copies one image on a worker and opens the
**Generate around head** window. The user marks a square covering the head and
describes the body and scene. `fix.around_head` runs a full-frame masked redraw
through a workflow's `face_detail` section; its noise mask excludes the marked
head. The original square is restored last at native size and coordinates.
Output retains the source frame dimensions. No face swaps or finishing passes
follow. Non-PNG sources are converted at full resolution before selection, so
the preview and generation use the same coordinate system. Unsupported formats
or workflows fail explicitly. This replaces the earlier likeness-only shortcut.

Identity experiments follow `docs/identity-recipe.md`. The CLI
`tools/identity_recipe.py` evaluates installed FLUX adapters in an isolated library:
portrait baseline/checkpoints at multiple seeds first, then scene changes only for
checkpoints with explicit passing reviews at every seed for the same recipe hash.
Render completion never certifies likeness. `recipes/partner-identity.json` records
the current pending trial. The ordinary Generate button is not gated by this tool.
`tools/prepare_identity_lora.py` remains a Partner-specific preparation script;
its captions/source ordering must not be reused for another person.

**Build LoRA** (identity editor, beside Use as primary; Sitter 2026-09-27: "a
simple build lora into the images of an identity. it needs at least 20 pics").
**It builds a FLUX.2 Klein 9B LoRA of the person's head, for the head swap -
not a FLUX.1 LoRA for the picture.** It trained Klein 4B until the head swap
moved to the 9B on 2026-10-01 (the user chose "9B + Build LoRA on 9B"); a 4B LoRA
does not load on the 9B, so a LoRA's family says which (`flux2-klein9b`, "FLUX.2
Klein 9B"; the 4B's were `flux2`) and `imagegen.head_lora` refuses a 4B one
with "rebuild it with Build LoRA". On the bench below, Partner's 9B LoRA built
by the app's own code (9.0 minutes, about 21 GB of the 5090) scored 0.723
in the head swap, 12 of 12 at 0.5 or more and the worst 0.56 - against the
9B alone 0.555 and her 4B LoRA 0.647 (worst 0.35); after the face swap 0.876,
the 4B LoRA's 0.875. By eye one consistent person on all 12; the stranger's
lighter wavy hair still survives on the dinner and beach pictures. The rest
of the numbers here are the 4B's
(2026-09-30, the user: "right now it's just
not a strong enough face swap", then "train a klein lora instead"). Measured
first, through the app's own head swap on 12 FLUX pictures of a stranger,
ArcFace against Partner's 23 real photos (her own photos score 0.80 against each
other): Klein without a LoRA drew a look-alike, 0.39 (3 of 12 at 0.5 or more);
with her LoRA, 0.65 (10 of 12), better on all 12; after the face swap 0.85 ->
0.875, the worst 0.78 -> 0.83 (inswapper is trained on ArcFace's features, so
those last numbers flatter it). Checkpoints scored alike from 500 steps to
1500 (0.645 / 0.658 / 0.634 / 0.621 / 0.647), so `STEPS` is 750: about 6
minutes on the 5090. The FLUX.1 build it replaced took 2.7 s a step (90
minutes) and reached no picture made with Z-Image.
`apps.image_studio.lora_train` (stdlib) refuses fewer than `MIN_PHOTOS` (20) photos that
exist on disk, saves the profile, and runs `tools/train_identity_lora.py` in
ai-toolkit's venv (`D:\ai-toolkit`) as a `core.procs` child, so closing the
app stops it. The script cuts each photo upright to a square round the head
(`head_square`: 2.4 faces wide, a little below the face's centre, made smaller
and slid inside the photo - padding it with white taught white bars), round
the person's own face (below: "The person's own face"; without the finder, the
biggest face OpenCV's cascades find, frontal then profile either way, and a
photo with none goes in whole), captions it "a photo of <trigger>" - so
anything else in the square, another face or a cut-out's white, is learned
as part of them - and trains Klein 9B's undistilled base
(`ARCH` flux2_klein_9b, rank 16, lr 1e-4, 512 px, bf16; only the text encoder
quantized, since Qwen3-8B in bf16 beside the 9B's 18 GB would not fit 32 GB
while the captions are encoded); the LoRA then loads on the distilled Klein
ComfyUI runs. `tools/prepare_klein_lora.py` sets ai-toolkit up once (in its
venv, with `huggingface_hub` and the saved token): Qwen3-8B from
Qwen/Qwen3-8B (16.4 GB, Apache 2.0; ComfyUI's copy is fp8, which ai-toolkit
cannot train with), Klein 9B's base from black-forest-labs/FLUX.2-klein-base-9B
(18.2 GB; gated by its own licence page, separate from the distilled 9B's - the
script says where to accept it), and ComfyUI's FLUX.2 VAE (diffusers layout: ai-toolkit
fails on `decoder.up.0.block.0.conv1.bias`) converted by ai-toolkit's own
`convert_diffusers_state_dict`. ai-toolkit's Klein class names the Hugging Face
repo for its text encoder, so the worker's `--aitk` mode starts run.py with
that pointed at the local folder and the hub offline: a missing file fails,
nothing is fetched. `problem()` names any missing file. ComfyUI's models are
freed first.
The 20 are checked again on the photos the script can actually read (the spec's
`min_photos`): an unreadable file used to be skipped as a note nobody saw, and
training went on with as few as one. It prints `KEPT n total`; fewer than
`min_photos` is an ERROR naming the unreadable files, before any training, and
any skipped photo is said in the tab ("warn"). The library record's "Built from
N photos" counts the kept ones.
Progress (`STEP n total`, parsed from tqdm) shows in the tab's note. The file
lands in the local backend's `lora_dir`, joins the library as an Identity
LoRA (family flux2-klein9b, its trigger on the record), and becomes the person's
`head_lora` - in the open editor's copy too, so a later Save keeps it ("Head
swap LoRA" in the editor). **Not `lora`**: an identity's `lora` joins every
picture's own LoRA stack and its trigger is said in the prompt; a Klein LoRA
is the head swap's alone. `Studio._head_swap` takes it through
`imagegen.head_lora` (in the library, the Klein 9B family, the file on the
backend - else Klein as before, and the record's notes say why) and
`headswap.head_graph` puts it on that head's Klein with its trigger in that
head's prompt (`prompt_for`); another person's head in the same picture is
drawn without it. The trigger is the profile's own, else `trigger_for(name)`
(`lilperson`). This assigns on completion, unlike the recipe tool; completion
is still not a likeness review. A build that dies any way at all still frees
the button (`finished` from a `finally`). Tests: `tests/test_lora_train.py`,
`tests/test_headswap.py`.

**The person's own face** (2026-10-01; the user: "crop as they import and fix the
trainer"). Partner's identity took 126 wedding photos of 6720 x 4480, mostly
group shots of ~10 faces. Rated by what Build LoRA did with each, 84 of the
101 then in her list failed on the trainer, not the photo: OpenCV's cascades
skip any face under a twelfth of the short side (373 px there), so 49 went in
whole with her face ~20 px of 768; and the biggest face it did find was
someone else's in 27 more. `apps.image_studio.faces` (stdlib) runs
`tools/identity_faces.py` in ComfyUI's venv (`STUDIO_COMFYUI`, default
`D:\ComfyUI`; InsightFace and antelopev2 are ComfyUI's, for PuLID - the
worker refuses a root without `glintr100.onnx` rather than let FaceAnalysis
download 360 MB) as a `core.procs` child. Who the person is: the mean ArcFace
embedding of their faces, seeded by pictures holding exactly one face, then
refined on each picture's best-matching face - a mean of every face in a
crowd is nobody. A face is theirs at `SAME` 0.42 or more (her 135 photos:
other people's best 0.34 at most, hers 0.53 at least). Faces read are cached
by file size and time in `face-cache.json` beside the library (embeddings as
base64 float16). Two uses:
- **Build LoRA** (`Build.find_faces`, when the spec has `face_cache`): before
  training, each photo's face as `[x, y, w, h]` in the spec's `faces` - the
  square the cascade would have drawn (`HAAR_SCALE` 1.27 of InsightFace's
  width, median of 103 faces both found), so `head_square`'s 2.4 is unchanged.
  The script cuts round that box; a photo with `null` does not show them and
  is left out (`LEFT n total`, said in the tab; a floor miss names them).
  Without the finder the tab says so ("warn") and the biggest face is used as
  before. Live on her 135: 62 s cold, 123 found and the 12 others left out,
  agreeing photo for photo with the ArcFace rating; the 83 the cascades had
  got wrong all now square on her.
- **Import** (`crop_for_import`, from `RecordEditor._import_paths` and
  `_add_character_photo`): each photo from disk is cut to the person's head
  and shoulders (`keep_box`: `KEEP_W` x `KEEP_H` 4 x 5 face widths, one above
  the face for the hair, slid inside the photo) before it is copied in; one
  they already fill (`KEEP_WHOLE` 0.8 of it) and one they are not found in
  stay whole, and the status counts each. The references are the person's
  other photos. Angles/Blend results are not cut (`crop=False`). Without the
  finder, photos go in whole and the status says why. Live: 6 wedding photos
  cut to ~700-1100 x 900-1400, a group shot where her face was too small to
  tell kept whole.
- **Rate photos / Remove duplicates** (identity editor, beside Build LoRA;
  The user: "a set of criteria that rates the images on best candidacy for lora",
  "i want the ui to show me the ratings also", "and remove duplicates"). The
  worker's `rate` mode adds what each photo's training square holds; the
  cache (`VERSION` 2) keeps per face its pose (`landmark_3d_68`), sharpness
  (Laplacian variance at 256 px wide), light and a dHash, and per picture its
  size and `kind` - "cutout" or "generated" from the ComfyUI graph in its PNG.
  `faces.score` (stdlib) rates 0-100: likeness 25, resolution 20 (the square
  against `TRAIN_SIDE` 512), sharpness 20, clean 15 (other faces in the
  square), light 10, real 10 (a cut-out's white and a Kontext edit's look are
  learned as the person's) - and 0 where the person is not found, as Build
  LoRA leaves those out. `faces.rate`: of each set of twins (`DUP_SAME` 0.95
  ArcFace - an Angles/variations picture beside its source - or `DUP_SIM` 0.90
  with face dHashes `DUP_HASH` 16 bits or fewer apart - a burst; ~0.88 is
  another moment of the day, kept) the best stays, the Primary always; `TOP`
  30 starred, best first with a bonus for a head angle the set has few of
  (`ANGLES`, left/right as seen in the picture). Each tile shows "★ 91" /
  "64" / "duplicate · 80" / "0 · left out"; hovering it puts `faces.detail`
  in the status. A rating then sorts the list (`faces.best_first`; the user: "it
  should auto sort them for the best pic to set as the primary"): Primary is
  the best-scoring front view (it is the face a picture is matched to), else
  the best of any angle; then the usable photos best first, the duplicates,
  the left-out ones - so this rating does not protect the old Primary
  (`rate(keep_first=False)`). Save keeps the order. A person's tiles are
  `IDENTITY_TILE` 220 px, unscaled ("can you make the thumbnails larger"):
  their previews are 220 px and Tk shrinks only by whole factors, so the old
  110-px tile showed them at half; the identity editor opens wide enough for
  four a row and `_refit_paths` redraws on a resize when a row fits more or
  fewer (`_path_cols`). Remove duplicates rates first if it must, asks, and takes
  the twins out of the list only (files stay; Save keeps it). Live on
  Partner's 135: 17 s cold, 1 s cached; 43 duplicates (by eye, all bursts or
  one face re-backgrounded), 12 left out, scores 57-100, 30 starred over all
  five angles.
Tests: `tests/test_faces.py`.

**Angles and Blend** (identity editor, beside Build LoRA; Sitter 2026-09-28: "add
breeding to reference images", then "breed is a seperate step. the angles are
something every photo has"; 2026-09-29: "rename breed to blend", so the button,
the window, the module and `blend_graph` are all Blend now and nothing is
called breed). Two buttons, two things. Angles: each selected
photo gets `NewPhotos.PER_PHOTO` (4) random, different `blend.ANGLES`, each a
single-image FLUX Kontext edit of that photo. Blend: exactly two selected photos
go in as chained `ReferenceLatent`s and the blend is drawn on an empty latent at
the first photo's shape (~1 MP), so it copies neither; each Again is a new
blend. `apps.image_studio.blend` (stdlib) builds the graphs, routes to an up
backend with `KONTEXT` (the 5090 first) and runs them through the studio's
client; `ui.NewPhotos` shows results as they come, and Add sends the picked
ones through `_import_paths` (Save keeps them). Nothing is scored: measured on
Partner, blends kept her (new settings), while angles are hit and miss. Profile
right and over-the-shoulder kept her; one profile left came out as a
short-haired look-alike, and three-quarter right barely turned. The person picks.
Refused while a LoRA build holds the GPU. Tests: `tests/test_blend.py`.

**Blend anywhere** (Sitter 2026-09-29: "put real infrastructure behind it"; asked
which - records and the queue, controls, likeness scoring, or blend anywhere - he
chose "Blend anywhere"). Blend is also a tool of the whole Image Studio, not only
of a person's references: any two pictures, of anything. It is a job mode like
Try On's, so it has no queue or record of its own. Settings are `{"mode":
"blend", "seed", "backend", "blend": {"images": [a, b], "person", "words"}}`;
`Studio.submit` and `Studio.run_job` hand that mode to `blend.submit` and
`blend.run_job` (imported inside the call: `blend` imports `imagegen`). The job
waits its turn in the backend's lane, shows Sampling → Decoding → Complete
(`blend.STAGES`), and is kept in History by `blend.record` with both pictures
under `references`, so it is in the Image library too, Generate Again remakes it
from its seed, and New seed is another. Its graph is its one `_run_pass`
("Blend"), so the record's `graph` is empty and the Nodes view shows it once.
`blend_words(person, words)`: with `person` the words are the identity editor's
(`BLEND` + `KEEP`), without they are `BLEND_PICTURES`, which keeps nobody - two
landscapes have no face to keep. `blend.problem` refuses, in words, fewer than two
pictures, the same one twice, and a file that is gone - at submit and again when
the job's turn comes. `route(studio, settings)` now honours a named backend and
`prefer_backend`. The window is `ui.BlendWindow` (one, raised when open;
`ImageStudio.blend(first, settings)`): two places, each filled from Library…
(`ImageLibraryWindow(pick=…)`, the library as a chooser) or File…, Swap, a switch
for the same person, and words to add. It opens from Blend… beside Fix a spot,
"Blend with…" on the picture's menu, Blend… in the Image library, and Reuse
settings on a blend. Each Blend is a new seed. The new picture takes picture 1's
shape. Not built, because he did not choose them: a balance between the two
pictures, what to take from which, more than two, and scoring. The identity
editor's Blend is unchanged: its results go to the person's references, not to
History. Tests: `TestBlendJob`, `TestWords` in `tests/test_blend.py`;
`test_blend_is_a_window_of_its_own_that_sends_a_job` in `tests/test_imagegen.py`.

Angles asks which way to look before it draws (Sitter 2026-09-29: "use angles as a
preset and ask which way we want it to look based on a cube like bambu studio has").
The window opens with a view cube (`apps.image_studio.viewcube.ViewCube`) instead of
starting a random round. The cube is the person: a face drawn on the front, and
RIGHT/LEFT are *theirs*, so seen from in front their right face is on screen left.
Each face is cut in three both ways, into 26 parts (6 face middles, 12 edge strips,
8 corners). Each part is a view key `(x, y, z)` in the person's frame: the camera
stands out along it. A click picks or drops a part; a drag (more than `DRAG` px)
turns the cube. `blend.view_name` names the key ("front right", "back left from
above", "straight below") and `blend.view_prompt` words it. Each prompt says where
the camera went *and* which edge of the picture they face, because "their left" and
the picture's left are opposite ways round: with the camera at their right they face
the picture's right. These are reference photos of a face, so back views have them
look back over the shoulder. The pick is the preset: saved on every change to
`angle_views.json` in `studio_dir()` (`load_views`/`save_views`; `DEFAULT_VIEWS`
when the file is missing or unreadable). Make draws every selected photo from every
picked view; Surprise me picks `PER_PHOTO` random ones. The old eight hand-worded
angles are gone. The cube's words carry their phrasing. **Live on Partner (2026-09-29,
all 26, one seed): Kontext turns a face only to the picture's left.** The right-hand
views came out the same as their left twins, so saying which edge they face does
not steer it. Above/below became a head tilt, not a camera height. Straight
above/below came out as the front view. Three-quarter views turned about 15
degrees. Front and back views worked. Flipping the photo (`ImageFlip`), asking for
the left twin and flipping the result back did give right-hand views that look like
her. So `angle_graph` now does that for every view from their right (`mirrored`, two
`ImageFlip` nodes, a core node both backends have). The words only ever ask for a
left view. Camera height took six live rounds. "Raise the camera high above this
person to a bird's-eye view … without changing their pose … their face pointing the
same way as before" gives a real high angle. "… their head tilted up towards the
camera" also did, but turned every view from above to face the lens. From above is
still the weak row: a turn keeps only ~30 degrees, and the side views come out
nearly frontal. "Rotate the camera
down to a worm's-eye view … zoom out so they tower over the camera … the ceiling and
ceiling lights" gives a real low angle ("from near the floor" and "the height of
their waist" did not). In one edit Kontext does *either* the turn *or* the height.
So a turned view from above or below is two edits in one graph (`view_steps`): the
level turn first, then the height on that picture (`KEEP_TURN`, seed + 1). Those
views take ~36 s rather than ~18 s. Front from above/below and straight above/below
stay one edit. Straight below comes out oddly posed (leaning over the lens) but is
seen from beneath. The final all-26 run on Partner: level and below right on both
sides; above high but mostly facing the camera. Tests:
`tests/test_viewcube.py`, and
`test_angles_asks_on_a_view_cube_and_keeps_the_pick_as_the_preset`.

Profiles' editable `description` is visual identity prose used alongside photos,
not the private `notes`. `identity_description_text` binds it to selected people
or Scene Builder's linked identities and positions; scene identities supersede
stale form selections. Keep the scene's clothes, pose and expression separate.
WithAnyone's optional `reference_mode="identity_consensus"` is an experimental
node path: primary SigLIP patches, pooled normalized ArcFace directions restored
to mean raw norm, one region per person. It is not the default and has not passed
The user's likeness review. `tools/validate_identity_guidance.py` exercises repo
code against installed vendor code without deployment; see docs/withanyone.md.
Profiles can opt in with `pool_photos` (People → Image references). The planner
uses `StudioWithAnyonePooled`, requires that node in preflight, keeps unpooled
people's primary photo only, and pools each enabled person's own set.
Scene-linked profiles carry the same setting. Never use the older repeated-token
experiment as a substitute for this switch.

The identity editor's **Add folder…** imports supported image files directly in
that folder (not subfolders), in filename order, on a worker. It keeps local
copies, deduplicates by image bytes against the existing set, and reports failed
files without dropping successful ones. **Use as primary** moves one selected
photo to the front; WithAnyone uses that photo, while FaceFusion uses the set.
Choose clear photos of the same person with one face per photo. Import checks
file signatures, not face quality or whether the photos show the same person.
Saving waits for pending imports. `tests/test_identity_import.py` covers copying,
duplicate handling, partial failures, primary ordering and the save boundary.

Selecting a profile in Image Studio's identity menu applies its saved reference
photos through FaceFusion after all generation and automatic refinement. The
profile's optional `avatar` is display-only (generated pictures are allowed);
it never becomes a face reference. `face_swap` defaults to true independently
of the older `use_references` switch. Profiles without photos retain their LoRA
behavior. The profile editor keeps technical fields under Advanced settings.

**A face FaceFusion will swap is not drawn with PuLID first** (2026-09-28,
The user: "this is a waste of time otherwise"). In a scene, the face pass
(`_face_pass`) used to give every scene person with a face photo a PuLID
likeness redraw regardless of whether FaceFusion was about to swap that same
person's face afterward - the swap fully overwrites the redraw's pixels, so
the likeness pass on that face was pure waste. `_face_pass` now asks
`facefusion.selected` which scene person ids it will swap (matched by the
scene's own `person_id`) and leaves those faces' redraw to words only, no
photo; FaceFusion still does the identity work, once.

`apps/image_studio/facefusion.py` is a stdlib adapter to the isolated environment in
`.runtime/facefusion-venv`; it uses `core.procs` for cancellation and process
containment. `tools/facefusion_swap.py` runs the official pipeline, captures its
mask, restores original pixels outside that mask, and checks the saved PNG.
Failure is explicit, never a silently substituted generated face. A face-only
Fix uses this same route. Reports are kept in history's `facefusion` field.
Reference previews are normalized off the Tk thread by `facefusion_previews.py`.
Tests in `test_facefusion_profiles.py` mock inference; the GUI tests exercise the
profile menu and editor. Partner's live final-pass result matched the approved
standalone HyperSwap result byte-for-byte (before the two changes below).

**The swap goes behind what is in front of the face** (`facefusion.SWAP_MASKS`:
box, occlusion, region; the worker's `--masks`; 2026-09-29). With box and region
alone the new face was painted over glasses frames - thin metal ones came back
mottled and half rubbed out, thick ones smeared at the bridge - which is why the
glasses pass exists. With FaceFusion's occlusion mask the frames are the
picture's own, to the pixel, on three pictures of three. ArcFace against
Partner's photos fell 0.02-0.04 (0.87 to 0.84): the glasses are then the
picture's and not hers. The report says which masks a swap used (`masks`).

**The person's averaged face is kept between runs** (`tools/facefusion_swap.py`:
`source_key`, `keep_source`, `kept_source`; `.work/facefusion-sources/`).
FaceFusion reads every reference photo for its face on every run, about a
second and a half each on the CPU: with Partner's 40 photos, 58 of a swap's 77
seconds. The average depends on the photos alone, so it is worked out once for
a set of them - keyed by FaceFusion's version and each photo's path, size and
time, in order - and read back after: 20 s a swap, and the picture is the same
file byte for byte (sha256, three runs). Adding, removing, replacing or
reordering a photo is another key; a kept face that cannot be read, or whose
first photo is gone, is not used and FaceFusion reads the photos as ever. The
newest `SOURCES_KEPT` (24) sets are kept. The report says `sources`: `kept` or
`read`. Two of her 40 photos have no face FaceFusion's finder sees, and three
score under 0.6 against the rest (a profile, a blue-lit one): the average
carries them all, as it did.

**A swap that fails says why, and one that is cancelled leaves nothing**
(`facefusion.failure`, `_clear`; 2026-09-29, from History). Of 57 pictures with
a face profile in three days, 7 failed and 6 were cancelled. Five failures
read only "[FACEFUSION.CORE] processing step 1 of 1": FaceFusion's content
check had refused the picture, which it does without a word. The worker now
watches that check (it is not changed, and a picture it refuses is not
swapped) and raises `REFUSED`. Two read as a 2,000-character traceback whose
last line was the reason - more than one face and no way to tell which -
and `failure` now gives the worker's last error in its own words. Three
cancels read "[WinError 32] ... run.log" and left their folders, the picture
in each, in the temp folder: the stopped worker still held its log while
`TemporaryDirectory` removed it, and that error replaced "Face swap
cancelled.". The folder is now removed after the worker is stopped, tried
again while Windows lets go, and its failure is never the swap's. **And the
target file's name is the swap's own** (`studio-facefusion-<random>.png`, the
folder's name): FaceFusion keeps its working copy under
`.runtime/facefusion-temp/facefusion/<target name>/` and clears that folder
before and after a run, so two swaps at once from two processes (two
sessions' scripts that day; in production the app and the phone server) with
the same `target.png` cleared each other's, and one ended "copying image
failed". `_SWAP_LOCK` only holds within one process.

**The swap is pushed past neutral, and drawn at the face's own size**
(2026-09-27, the user: the face swap "isn't as strong as i'd like"). Two FaceFusion
settings had been left at their defaults. `--face-swapper-weight` (FaceFusion 3.4+)
was 0.5, which hands HyperSwap the references' identity as it is; above 0.5 the
identity is extrapolated away from the face being replaced, so less of the
generated person survives. `apps.image_studio.facefusion.SWAP_STRENGTH` is 0.8, and a
profile's own **Face swap strength** (`swap_strength`, 0-1, shown beside the Final
face swap switch) wins; FaceFusion only takes multiples of 0.05, so `strength()`
rounds (and since 2026-09-29 holds it to the swap model's peak, `SWAP_PEAK`:
see "With inswapper a strength over 0.5 is held to 0.5" below). And `--face-swapper-pixel-boost` was 256: HyperSwap draws at 256 px, so a
bigger face came back as a soft 256 px face scaled up. `select_target` now sets
the boost to the smallest size at least 1.5x the chosen face's box (its warped
crop is about that), up to 1024; the report records `weight` and `pixel_boost`.
Neither has been measured live yet. The eye pass that follows (0.5) redraws the
eyes from words on the picture's model, and eyes carry much of a likeness: if a
swap looks right before the finish passes and weaker after, that is the place.
Measured on 2026-09-29 (two pictures): the eye pass costs 0.03 of ArcFace
likeness, the glasses pass after it another 0.08 - see the glasses below.

**The head is redrawn before the face is swapped, and the swap is inswapper's**
(2026-09-29, the user: "i think we need flux klein", then "klein with inswapper seems
to work well enough"). FaceFusion changes the inside of a face and nothing else, so
the head it landed on - shape, hair, glasses - stayed the generated stranger's.
`Studio._head_swap` (Generate only, passed to `finish_profiles` as `before`, so the
checkpoint kept before the faces is still the picture as generated) runs
`apps.image_studio.headswap`: one SAM3 run (`FIND`, faces), `targets` picks each
profile's face by FaceFusion's own `target_face` rule, `head_crop` cuts a square
`CROP` (4) faces wide round the face, and `head_graph` has
FLUX.2 Klein 9B (fp8) redraw it at 1024 px from an empty latent with two
`ReferenceLatent`s - the crop, then the profile's FIRST photo - in 4 steps at cfg 1
(ComfyUI's own Klein template). Its colour is moved `TONE` (0.5) of the way to the
crop's (ColorTransfer). It is blended back through the head and hair alone:
SAM3's `WORDS` (`head`, `hair`, `necklace`; bare words, see "The old hair is taken off
whole" below) in the crop before and after, grown, inside a soft square. **Klein 9B since 2026-10-01** (the user: "id would like to try flux
klein 9B", then the 4B's weights removed): on the 12 stranger pictures of
Partner's LoRA bench, with no LoRA, the 4B scored 0.386 (3 of 12 at 0.5 or more)
and the 9B 0.555 (8 of 12); after the face swap 0.849 and 0.860. By eye the
9B keeps the picture's expression, clothes and light where the 4B pasted in
the photo's smile and top. About 8 s a head on the 5090. Its files:
`flux-2-klein-9b-fp8.safetensors` (black-forest-labs/FLUX.2-klein-9b-fp8,
gated), its own encoder `qwen_3_8b_fp8mixed.safetensors` (Comfy-Org/flux2-klein-9B;
CLIPLoader type `flux2`) and the shared `flux2-vae`. **The 9B is under the
FLUX Non-Commercial License, so every picture it touched says so** (the user:
"attach a note to images"): `Job.license` = `headswap.LICENSE_NOTE`, kept as
the record's `license`, shown on its History row, and written into each PNG
as a tEXt `Comment` by `History.add` (`png_text`), so the terms go wherever
the file goes. `settings["head_swap"]` (on by
default and for pictures saved before it; "Head swap before the face swap" under
Generate) turns it off. A backend without SAM3, Klein's three files or its nodes,
a failed run and a cancel all leave the picture as it was generated, with a note:
the head swap is an extra and never costs the picture. The local face-only swap
(Fix a spot, Retry face swap) has no ComfyUI and gets none.
`facefusion.SWAP_MODEL` is now `inswapper_128` (a profile's `swap_model` wins;
HyperSwap 1a was the first recipe). Measured on Partner, ArcFace against her photos,
on the two pictures of the FaceFusion and BFS trials: no swap 0.11-0.17, HyperSwap
1a 0.20 and 0.70, Klein alone 0.23-0.35, inswapper alone 0.81 and 0.78, Klein then
inswapper 0.77-0.85, the same at strength 0.5, 0.65 and 0.8. inswapper was trained
on the ArcFace that scores it, so its numbers flatter it. Live through
`Studio.submit` by script on the 5090 (Z-Image, one seed, a dancer in a meadow): 0.70
with the head swap against 0.57 without, 108 s against 94 s, the head swap itself
about 8 s. That first version left two flaws the user asked to have fixed ("fix the
two flaws first"). **The generated hair stayed**: long blonde hair under a darker
crown. The mask was not the cause - Klein itself drew image 1's hair, on every
seed, until `PROMPT` told it to remove that hair "including any of it lying on
the neck, shoulders and chest"; the crop grew from 3 faces to 4 and lost its
rise so that hair is inside it, and `hair:1` joined the mask so what Klein took
off does not come back from underneath. Dark, pulled back, on 3 seeds of 3.
**The face came back pale pink in a low sun**, from two places. Klein's head:
the prompt now says whose light falls on it. FaceFusion's swap, which paints
the references' skin over a head that had the light right:
`tools/facefusion_swap.py --tone` (`facefusion.SWAP_TONE`, 0.8) moves the new
face, inside its mask, to the Lab mean and spread of the face it replaced, the
spread's gain held to 0.7-1.3. It applies to every FaceFusion swap, with or
without a head swap. Live after both: 0.71, 104 s. Still open: where the old
hair lay Klein draws skin and blouse of its own, and into that it copies things
from the photo - a thin necklace on 3 seeds of 3, a floral trim on the blouse on
1 of 3; the prompt's last sentence lessens it, and cutting the photo to its head
(`photo_crop`, which `head_graph` takes and `_head_swap` does not yet send) did
not help with a photo that is mostly head already. The colour match is a modest
change, not a relight; a cheek keeps some pink. The seed matters: one close-up
scored 0.23 and 0.35 on two seeds. Tests: `tests/test_headswap.py`.

**The old hair is taken off whole** (2026-09-29, seen live: pale wisps of the
generated wavy hair in the air round a head whose hair Klein had pulled back).
Three causes, found by saving the head swap's masks and Klein's own crop beside
its result, and one thing the repair brought with it.
- **SAM3 was not asked for the hair.** ComfyUI's SAM3 encoder
  (`comfy/text_encoders/sam3_clip.py`) takes the count off a word when there are
  several words or the count is over one - `face:8` is "face", up to 8 - and
  hands a lone `hair:1` on as that text, colon and digit and all. SAM3 answered
  "hair:1" with the whole person on one picture, the glasses on a second, the
  glasses and a wine bottle on a third; bare "hair" is the hair on all of them.
  So the blend went through the whole body where it found the person (the
  sweater was Klein's, the pendant gone) and through the head alone where it
  found the glasses. `WORDS` are bare: one is the count of a bare word already.
  **Never write `word:1` for SAM3.**
- **Loose strands lie outside SAM3's hair.** The old hair - `OLD_HAIR`, the
  hair in the crop BEFORE, not Klein's - is grown `STRANDS` (48 px at the
  crop's 1024) and round (`tapered_corners` off: GrowMask's tapered growth is a
  diamond, half as far to a corner), and joined to the rest. 32 px left the
  furthest strands, 48 none on 5 pictures of 5, 64 began to take a pendant.
- **The soft square kept the top of the old hair in a close-up.** A face a
  quarter of the picture wide makes the crop the whole picture, and the
  square's margin (`EDGE`) then ran along the picture's own edge, under the
  crown. `head_graph(size=)` is the picture's width and height, and a side of
  the crop that is the picture's edge has no margin.
- **A necklace under the old hair was cut.** Through the head and hair alone
  the body is the picture's again, and with it a pendant on a cord that ended
  where the old hair began: `PROMPT`'s "no ... necklace or jewellery taken from
  image 2" had Klein take image 1's own necklace off too (9 runs of 9), and
  copy the photo's floral top all the same (2 of 9). It now says first what
  image 1 keeps ("the necklace or jewellery image 1 wears, if any"), then what
  is not copied: the cord whole on 9 of 9, nothing of the photo's on 9 of 9
  (3 pictures x 3 Klein seeds). A wording between the two left a second
  necklace and a floral top on 1 of 9 each. Klein's cord is a few px off the
  picture's, and where the old hair ended the cord forked (2 pictures of 5):
  `necklace` is the third of `WORDS`, so a necklace is Klein's from end to
  end, and `REACH` takes it `CORD` (24 px) round, before and after. At `GROW`
  the blend's soft edge ate the thin mask from both sides and the picture's
  own cord showed faintly beside Klein's.
Klein may draw a passer-by near the old hair sharper than they were, and
FaceFusion's own finder then sees two faces where it saw one and will not
choose (a cafe, live). `_head_swap` keeps the middle of each face it redrew
(`job.heads`, `headswap.middle`), and `_apply_profiles` points the swap at it
(the profile's `target_point`, on a copy; a profile with a region or point of
its own keeps it). Still open, and Klein's, not the mask's: on some seeds it
leaves the face the generated one (ArcFace 0.09 on one of five) or draws the
person with long hair (one of five); the swap after it still carries the
likeness there (0.79 and 0.82 at the end). And SAM3's head has small holes
at the eyes behind glasses on one picture of five, where the blend is the
generated picture's; the swap and the eye pass draw over them.

**No ghost of what the old head wore** (2026-10-01, the user: "fix the flower
crown ghost"). Partner's profile carries "flower crown" from earlier pictures;
Klein drew her head without it, and a grey crown stayed round the new head -
on 8 runs of 8 with her head LoRA (2 pictures x 4 seeds). Found by saving each
stage of the head swap (crop, Klein raw and toned, every mask): the
composite is exactly crop x (1 - mask) + Klein x mask, so the ghost was two
things.
- **The crown lay outside SAM3's head and hair**, and only the blend's soft
  edge reached its outer flowers, which came through faded. `WORN`
  ("headwear") is masked before and after; SAM3 masks a crown alike for
  "headwear", "hat", "flower crown", "tiara" and "headband". On the 12 bench
  pictures, where no head wears anything, it also found hair, a lace collar
  and a lily, so it counts only within `WORN_NEAR` (half a face, crop side /
  `CROP`) of that same picture's head and hair. Before, the old crown goes
  whole; after, a hat Klein draws from the photo is not cut at the hair line.
  On the bench it changes no pixel.
- **The soft edge showed the mask's own outline.** Klein's background is a
  few levels off the picture's; with the crown in the mask, the edge ran
  round the crown and its shape still showed on the dark window. The blur was
  26 px (ImageBlur stops at radius 31, sigma 10). Now the mask is grown
  `FEATHER` (0.08 of the crop's side) first, so everything it covered is
  still Klein's, then blurred over as far again at 1/`SHRINK` of the size and
  scaled back. 0.05 still left a faint outline on the dark side; 0.08 none,
  on 8 of 8.
Bench with her 9B LoRA (12 pictures, ArcFace): today's main 0.723 before the
face swap / 0.876 after; the identity-finish merge alone 0.594 (three heads
at 0.30, the narrow edge keeping part of the generated face) / 0.873; with
both of the above 0.688 (12 of 12 at 0.5 or more) / 0.886. The cost: 32% of
a picture changed on average against 27%, most in close-ups, where the crop
is the whole picture and the wider edge lets more of Klein's clothes through
(the floral top her LoRA brings from her photos). Seen on the bench and not
from this change: the identity-finish merge adds a thin necklace Klein drew
on the dinner picture, which today's main does not.

**The swap is a finish on the head: no weave, no teeth, no cheek behind a lens**
(2026-09-29, seen live: a pink patch on a cheek and a strained, yellowed smile
where Klein's head had looked natural). Looked at three times enlarged, the
swapped face had three faults, each inswapper's and each with its own cause.
- **The weave.** Pixel boost swaps a face bigger than inswapper's 128 px as
  N x N faces of 128, each every Nth pixel of it, and weaves them back. The
  model does not draw them quite alike, so the weave shows as a comb of
  streaks N px apart down a cheek and a grid behind the glasses, at every
  boost over 128 (256 to 1024 tried). `tools/facefusion_swap.py` `even`
  (`--deweave`, `facefusion.SWAP_DEWEAVE` 1.0) averages the woven face over a
  box as wide as the weave, which takes out what repeats every N px and
  nothing coarser: ArcFace 0.887 before and after. Boost 128 has no weave
  and is soft (0.859).
- **The teeth.** Drawn at 128 px they came back as yellowed blocks.
  `facefusion.SWAP_REGIONS` is FaceFusion's face mask regions without `mouth`
  (the inside of it; the lips are swapped): the teeth stay the picture's.
  0.887 with them swapped, 0.886 without. Without the lips too, 0.876.
- **The cheek behind a lens.** inswapper paints a bare cheek where the picture
  has one seen through a lens: a pink patch with a hard edge under each eye.
  Behind glasses the eyes are swapped and what lies below
  `facefusion.SWAP_LENS_LINE` (0.47 down the swap's own crop, where the face is
  upright, the eyes at 0.40 and the tip of the nose at 0.56) is left
  (`under_lenses`, `--lens-line`; the glasses are FaceFusion's own face
  parser's). Leaving all that is behind the glasses cost 0.05 (the eyes are
  most of a likeness), the line 0.01-0.03; at 0.44 it cut the lower lids.
  It applies with or without a head swap.
Not the cause, tried and left: the colour. Matching the swap's colour and
light to the head's at finer scales (the mask's size / 6 to / 48) changed the
patch by nothing one can see - it has an edge, it is not a cast - so `--tone`
is as it was. Putting the head's fine grain back on the swapped skin looks
better (pores, freckles) and cost 0.03-0.04, with a ghost of the head's own
brows: not done. FaceFusion's `face_enhancer` (GFPGAN 1.4, downloaded at
Sitter's word 2026-09-29, `.runtime/facefusion/.assets/models/`) is wired
and off: see "The face enhancer is there and off" below. The worker prints
the faces it found ("Faces found, left to right"), and the report says
`regions`, `lens_line` and `deweave`. A FaceFusion without
`create_region_mask` or `explode_pixel_boost` under those names swaps as it
does and the report says 0 for what was not done.

**With inswapper a strength over 0.5 is held to 0.5** (`facefusion.SWAP_PEAK`,
in `strength()`; 2026-09-29). This reverses, for this model, what the user asked
for on 2026-09-27 ("isn't as strong as i'd like", above) - the model then was
HyperSwap. FaceFusion's weight 0.5 hands the model the person's face as it
is; above it the face is pushed past theirs, away from the face replaced, to
take the last of the stranger out. With inswapper it took the person out.
ArcFace against Partner's photos, the swap alone, same picture and settings but
for the weight: on five heads Klein had drawn 0.848 at 0.5, 0.831 at 0.8,
0.813 at 1.0; on two generated faces with no head swap 0.836 at 0.5 and 0.798
at 1.0 - lower at the higher weight on every one of the seven, in every arm
tried (as shipped, without the weave and teeth, with the lens line). To the
eye the two are near alike, the higher a little harder. A profile keeps the
number it was given (Partner's says 1.0) and a strength under 0.5 is as it was;
the editor's label says what the number does. Any other swap model has no
peak: none was measured. If the user wants it back, `SWAP_PEAK = {}`.

**The face enhancer is there and off** (`facefusion.SWAP_ENHANCE` '' /
`SWAP_ENHANCE_BLEND` 60, `enhancer()`; the worker's `--enhance` /
`--enhance-blend`, `through`; 2026-09-29). The user asked for GFPGAN to be
tried at a low blend, and allowed the download (gfpgan_1.4.onnx, 340 MB,
from facefusion-assets `models-3.0.0`, crc32 checked as FaceFusion checks
it). It runs after the swap on the same face (`captured['target']`), and its
change is taken through the swap's own soft mask, so the teeth, the cheek
behind a lens and all outside the swap stay the picture's, and "zero pixels
outside the mask" still holds. What it does: sharpens the eyes, lashes and
lips; the skin stays smooth (it does not bring pores or freckles back).
ArcFace, the swap alone on the five heads: 0.846 without, 0.839 at blend 20,
0.832 at 40, 0.818 at 60, 0.802 at 80. At the end of the pipeline the eye
pass redraws the eyes it sharpened, and little shows for the cost: 0.812
without, 0.797 at 40, 0.786 at 60. So it is off. The enhancer is only ever
run when its model and hash are installed, never fetched by the app.
`MODELS` is FaceFusion's model folder.

**After the enhancer there is no eye pass** (`eye_pass`, in
`_finish_passes` and `pipeline_stages`; 2026-09-30, the user: "try gfpgan 60
without the eye pass"). The eye pass is there for the soft eyes a swap
leaves; after GFPGAN it cost 0.03 for eyes no sharper, so a swap whose
report names an enhancer (`enhance`) gets none, with a note, and the face is
not looked for unless the glasses are to be redrawn. The trial itself, end
to end on the five cases on main's redraw (beta): the swap then the eye
pass 0.813 (0.815 / 0.813 / 0.789 / 0.826 / 0.821), GFPGAN at 60 and no eye
pass 0.818 (0.830 / 0.827 / 0.786 / 0.829 / 0.819) - the same likeness, ten
seconds sooner, one pass fewer to fail (the eye pass failed once in these
runs on a ComfyUI three sessions were using: "hostbuf_file_reader_read
failed", and the picture was kept as the swap left it, as it should be).
Looked at, neither wins. The eye pass draws clear, bright eyes and gets
their colour wrong on 2 of 5 (brown where Klein had drawn blue-grey: its
prompt, `EYE_WHAT`, says nothing of the colour). Without it the eyes are
inswapper's behind the glasses, sharpened: darker, softer, heavy-lidded, in
Klein's colour on 3 of 5 and one eye darker than the other on 2. So the
default stays the eye pass with no enhancer. If the eye pass is to be
bettered, tell it the person's eye colour.

**What the two repairs come to, end to end** (2026-09-29, `Studio.run_job` on
a copy of the library, Partner's job of 15:09 on 5 seeds: three before a plain
wall, a park, a busy cafe; every picture looked at whole and at the face).
Before: the old hair left on 5 of 5 (loose strands on 3, whole locks on 2),
the pendant gone on 2 and a floral top from her photo on 1, teeth in
yellowed blocks or a ragged line of them between the lips on 5, pink under
the lenses on 5. After: none of these on any of the five, and the necklace
whole on 5; on one Klein drew her hair long and on one left a few loose
strands of its own by the ears. ArcFace against her photos, mean of the
five: generated 0.07, Klein's head 0.32, the swap 0.85, the end (after the
eye pass) 0.81 - against 0.83 at the end before. The 0.02 is the lens
line's; the strength held at 0.5 gave back 0.03 (0.79 at the end with her
profile's 1.0 handed on as it was). The eye pass costs 0.03, as it did, and
is not changed here. 33-87 s a picture on a free card, as before.

**The eyes and then the glasses are redrawn after the swap** (Generate only;
`Studio._finish_passes`). FaceFusion pastes the new face over the frames,
so the eyes came back soft and the glasses faint. The user asked for "an eye pass
and glasses last" (2026-09-26). The swapped picture goes back to the job's
ComfyUI. One SAM3 run (`FINISH_FIND`: faces, glasses) finds the faces, and
`swapped_faces` picks the ones FaceFusion swapped, by its own `target_face`
rule over the faces left to right. With one profile it falls back to the
biggest face. Two fix-machinery runs of `face_graph` follow, on the picture's
own model. First the **eyes**: the whole face is cropped and only SAM3's
`eye:2` inside the face's eye band is redrawn (`eye_spots`, 0.5). Then the
**glasses** on that face, last, so nothing is drawn over them (`glasses_spots`,
0.45). Neither pass tone-matches, because the curves posterize swapped skin.
Live on Partner: at 0.6 the frames came back crisp but the lenses went milky
over the new eyes. At 0.45, with glare-free lenses in the prompt, the eyes show
through. Both runs take about 13 s on the 5090. Without SAM3 or a `face_detail`
section, and on any failure or cancel, the picture stays FaceFusion's and a
note says why. The local face-only swap (Fix a spot with no spots, Retry face
swap) has no ComfyUI and gets neither pass.

**The glasses are redrawn only when the swap painted over them**
(`glasses_pass`; 2026-09-29). The pass answers a fault the occlusion mask
(above) no longer makes. Run all the same, after a swap that had kept the
frames as drawn, it gave the person other glasses - dark red frames came
back tortoiseshell - and cost the likeness 0.08 (ArcFace 0.81 with the eye
pass alone, 0.73 with the glasses after it, two pictures, the redraw sampler
either way). So the glasses are looked for and redrawn only after a swap
whose report names no occlusion mask, and a note says when they were left.
The user asked for the pass by name, so it is his to have back: "Redraw glasses
after the face swap" under Generate (off; ticked is
`settings["glasses_pass"]` True, always; unticked is None, the rule above;
False never). The eye pass stays: it costs 0.03 and the eyes are sharper for
it.

**A Generate of people then has its hands redrawn** (the hands pass, same
method). The user asked for "a pass with natural hands" to go with the glasses
(2026-09-26). It runs with or without a face profile, unless "Natural hands
pass" under Generate is unticked (`settings["hand_pass"]`, on by default and
for pictures saved before it). `HAND_FIND` joins the same SAM3 run. Each hand
is a `found_spots` square grown by `FIX_CONTEXT`, and only SAM3's `hand` inside
it is redrawn, all the hands in one `face_graph` run, with `HAND_WHAT` ("four
fingers and a thumb"). The reverted per-hand ControlNet pass of 2026-09-25 got
double hands at 0.85-0.9. The order is eyes, hands, glasses. Hands tone-match
(`FIX_TONE`) when the ComfyUI has `StudioMatchTone`; a swapped face does not.
With no hands found, a note says so and the picture is kept. Three things
changed on 2026-09-29, each from pictures made that day:
- **It is a finish, at `HAND_DENOISE` 0.4.** It ran at 0.6, the Critic's
  strength for a hand that is wrong, on every hand - and Z-Image Turbo draws
  most hands well. On hands that were fine, 0.6 took the ring off a finger,
  aged a florist's hand into scales, bent a guitarist's fingers off the frets
  and gave a knitter a scabbed knuckle; on a small blurred hand in a crowd it
  did the good it was meant to. 0.4 with the redraw sampler left every good
  hand as it was and still sharpened the small ones. A hand that is wrong is
  the Critic's, or Fix a spot's, at their own strengths, unchanged.
- **The hands are SAM3's surest** (`real_hands`, `parts_found(scores=True)`).
  SAM3 scores each box, and asked at threshold 0.3 it also gives the forearm
  round a hand (0.33-0.57), a thing a few pixels wide (0.65) and the field a
  picture is of (0.48); a hand in plain view scores 0.78-0.97. `found_spots`
  takes boxes biggest first and drops a box mostly inside a kept one, so the
  forearm won and the hand in it was dropped as its copy: a carpenter's two
  hands were redrawn as two forearms and a third thing. They are now taken by
  score (`HAND_SCORE` 0.5 or more), a box sharing half the smaller with a
  better one is that hand again (`HAND_SAME`), and one under `HAND_SMALL` of
  the picture is a passer-by's.
- **It is for pictures whose words name a person** (`hand_pass`, `has_person`).
  SAM3 scored a red fox's paws 0.87 as hands, and the pass redrew them as "a
  human hand with four fingers and a thumb"; on an empty field it redrew the
  field. No score tells a paw from a hand (SAM3 scored the fox 0.8 as a
  person too), so the words decide, as they do for the clothing floor. A
  note says when it was left out for that. `PEOPLE` therefore knows a person
  by their trade and family as well ("a chef plating a dish" names no man or
  woman) - only words that can mean nothing else: no "baby", "model" or
  "player".

### Family photos with WithAnyone

`comfy_workflows/withanyone.json` and `comfy_nodes/studio_withanyone` draw one to
four identities together. `plan_identities` uses Scene Builder's `scene_faces`
photos and normalized head regions, or selected identity profiles placed left
to right. A missing photo or invalid region is an error, not a stranger drawn in
their place. The node refuses reference pictures containing multiple faces.
`multi_identity` on the workflow adds optional face inputs in `fill`, bypasses
PuLID injection, disables the face pass and skips the automatic critic redraw.
It also bypasses FaceFusion validation/swapping and all finishing passes, even
when selected identity profiles have `face_swap` enabled. A multi-photo experiment
used all unique library photos (up to eight per person), including linked Scene Builder
identities. The user rejected its result as not looking like Partner (2026-09-27).
It is now disabled by default: the normal form uses the first photo and says so.
Only `experimental_reference_groups` enables it for developer comparisons. Never
present passing execution tests as likeness validation. The library is preserved.
`StudioWithAnyoneReferences` groups extra images without resizing;
`references.py` flattens each group with repeated target regions, so upstream's
attention mask lets every view guide the same person. This is an experimental
multi-reference extension, not upstream's default. Missing files and ambiguous
source faces fail explicitly; History retains every photo. Install both node
Python files and restart ComfyUI; old nodes fail grouped-job preflight.
Scene Builder's `takes` includes `face_positions` for this workflow: no pose or
depth map is sent, and the UI says that poses and props come from words.

The node wraps pinned upstream WithAnyone code installed by
`tools/install_withanyone.py`. Only ComfyUI imports its ML dependencies. The
wrapper owns the diffusion model for one call, checks cancellation at each step,
and releases the model in `finally`; ComfyUI's `/free` cannot free upstream
models that bypass its model patcher. T5, SigLIP and face detection run on CPU;
VAE decoding is tiled. The likeness is `siglip_weight` (1.0; ArcFace gets
`1 - siglip_weight`): upstream's "Resemblance in Spirit <-> Form" slider, whose
demo default is 1.0. The port's 0.8 was weaker than the user wanted (2026-09-27);
see `docs/withanyone.md` before lowering it. Initially only the 32 GB 5090 is enabled. SigLIP folder
readiness comes from the node's `/object_info` choices because `/models` lists
files, not those directories. See `docs/withanyone.md` and
`tests/test_withanyone.py`; `tools/try_withanyone.py` exercises the real job path.

### The Visual Critic: automatic refinement after a picture is made

"Automatic refinement" under Generate (`auto_refine`, off by default) runs
`Studio._refine` on the lane's thread after the picture and its face pass:
the host's vision model (`Chat.vision`, handed to `Studio` as `vision`) looks
at it, and what it finds wrong is redrawn, up to `refine_passes` (3) times.
`apps/image_studio/critic.py` is the logic, no tkinter and no I/O of its own. The rules:

- **Three states, kept apart.** The *intent* (`intent_from`: the composed
  prompt and the form's words) is a read-only mapping; nothing writes into
  it. The *canonical state* (`initial_canonical`: character_a from the look
  slots and identities, the scene, the camera and style) holds one value per
  key; every key the form set is `locked`. The *corrections* are rebuilt from
  each look and never accumulate.
- **The critic returns JSON, every observation with a status** (MATCH,
  MISMATCH, UNCERTAIN, NEW_USEFUL_DETAIL), a confidence and a severity.
  `clean_result` makes anything malformed safe: an unknown status is
  UNCERTAIN, never a fix. No JSON at all keeps the picture, with a warning.
- **The planner prefers the smallest edit** (`plan_next_refinement`,
  `action_for`). A *major* scene/camera/lighting/body fault is the only
  thing that makes a new picture (`_regenerate`: `prompt_override` in
  `compose`, a new seed), and then it is the only action. Otherwise faces
  (FACE_CORRECTION), hands and small things (LOCAL_INPAINT,
  OBJECT_CORRECTION) run together, and a whole-picture touch-up
  (GLOBAL_REFINEMENT, denoise 0.2) waits until nothing local is left.
  Mismatches under `MIN_CONFIDENCE` (0.6) are logged and ignored. The loop
  stops when the critic says nothing meaningful is wrong.
- **There is no inpainting node here; the face pass is the local editor.**
  Every local fix is `face_graph` - SAM3 finds the thing (`add_face_finder`
  on a `LoadImage`, prompt `face:8`, `hand:4`, `<object>:4`), each box is
  cropped, redrawn with the job's model and LoRAs from a prompt compiled for
  it, and blended back through the soft oval. A crop with `mask` False and
  its own `edit` size is the whole-picture pass. So anything outside a
  crop is untouched pixel for pixel, which is what "preserve" means here;
  "keep X" in a FLUX prompt at CFG 1 does nothing. Hands stay at denoise 0.6
  (`CRITIC_DENOISE`) for the reason in the hand-pass note: higher left
  double hands. Only a template with a `face_detail` section (today
  `flux_dev_baseline`) can take a local fix.
- **Good inventions are promoted, once.** A NEW_USEFUL_DETAIL at 0.75 or
  more, on a key the user did not set and not also called a mismatch, is set
  at its key (`merge_canonical`; "Hair colour" and "hair_color" are one key)
  and goes into the next prompt. **And into every later picture:** each
  promotion is filed in `image-studio/critic_memory.json` (`remember`) -
  a person's detail under the identity's id (only when the picture had
  exactly one identity), a scene's under its words (`_key`'d). `compose`
  appends what `recall` finds for this job's person and scene to every
  prompt, auto-refine on or off, skipping any key the form sets, and
  `_refine` seeds the canonical state with it (unlocked) so the critic
  checks those details instead of inventing new ones.
- **Two outputs from one compiler.** `build_refinement_instructions` writes
  the sectioned text (ORIGINAL USER INTENT, CANONICAL ..., PRESERVE,
  CORRECT) to the log; `generator_prompt` is the prose FLUX reads, with a
  close-up head for a crop.
- **Every pass is scored by the next look** (2026-09-29). What a pass redrew
  are its *faults* (`as_faults`: numbered, `tries` counted); the next
  question asks after each by number (`FOLLOWUP_PROMPT`) and the answer's
  `followups` say CLEARED, PERSISTS, WORSE or UNKNOWN (`score_fixes`; a fault
  the model skipped is read from its observations by name, else UNKNOWN -
  never a guess). PERSISTS goes again *harder* (`critic.harder`: +0.15 a
  try up to `CRITIC_DENOISE_TOP`; a hand's top is its 0.6, so its second try
  is a new seed and the critic's newer words), with the followup's
  `correction` as the fix. After `MAX_TRIES` (2) it is `left` to the user and
  never planned again, even if the critic names it again (`closed`). A pass
  with a WORSE and no CLEARED is taken back (`went_wrong`): the files before
  it are the picture again. So there is one look more than passes - the
  last pass is looked at too, with no redraw after it. This is the only
  thing that carries from look to look; the corrections still do not.
- **The user's notes are faults too** (Fix a spot's Wrong field and
  "Critic checks it": `fix["check"]`, a spot's `note`, the fix's `note` for
  spots without one). `run_fix` redraws the spots as always, then hands
  `_refine` `marked` - `critic.user_faults` (confidence 1.0, tried once),
  `redo` (the fix's own `redraw` closure, on some spots, each with its own
  prompt and denoise through `face_graph`'s `faces[i]["prompt"]`) and the
  spots' crops. Then only the marked spots are followed: what else the
  critic sees is logged, never redrawn, since a fix changes only its
  squares. The critic is shown up to `CLOSEUPS` (3) spots cut large
  (`_closeups`: ImageCropV2 -> PreviewImage) *instead of* the references;
  without them it judges by the whole picture. A note says what is wrong,
  for the critic; Describe says what to draw, for the model. The note never
  goes into a prompt, and the critic never rewords it.
- **The scores add up in a ledger** (2026-09-29):
  `image-studio/critic_ledger.json`, pure logic in `critic.py`, read and
  written by `Studio._learn` under `CRITIC_LEDGER_LOCK`. Three things are
  filed. *Fixes* (`note_fixes`): model -> kind of fault -> `ACTION@denoise`
  -> tried / cleared / worse, one entry per scored redraw; UNKNOWN files
  nothing. *Faults* (`note_picture`): per model, per person (only a
  picture of exactly one identity) and per scene, each kind's `weight` -
  +1 when the critic found it, +`USER_WEIGHT` (3) when the user marked it,
  -1 for every picture the critic looked at without finding it, capped at
  `WEIGHT_TOP` - so a fault that stopped coming back is unlearned. *Blind
  spots* (`note_blind`): what the user marked on a picture whose record
  says the critic passed it. A **kind** is `category/target`
  (`fault_kind`: "realism/hand"), never the critic's wording, which
  differs every time; that and the caps (`KINDS_MAX`, `BLIND_KEPT`,
  `MARKED_KEPT`) are what keep the file from growing. The critic is a 7B
  and misreads: what it says must repeat (`RECUR` 3, `LEDGER_MIN` 3 tries)
  before it changes anything, what the user marks counts at once.
- **The next picture starts from the ledger.** Four uses, each said in a job
  note when it acts. (1) `start_denoise`: a redraw starts harder when its
  default strength mended under half of 3+ tries on this model and a
  harder one, within `CRITIC_DENOISE_TOP`, mended half or more
  (`_critic_denoise`; a fix's strength stays the user's). (2) `recurring`
  -> the critic's question gets WRONG BEFORE, the returning faults to look
  at first (`FIRST_PROMPT`; not in a fix's check). (3) `prevention` ->
  `compose` appends the critic's newest fix for a returning fault to the
  prompt, but only for one person's faults of a `PERSON_BOUND` category
  (identity, body, clothing): "a broad square jaw" describes her, "five
  fingers" describes no one and the anatomy text says it already. (4)
  `blind_checks` -> "; missed before: ..." on that category's line of
  `CHECKS`, the three most missed.
- **Fix a spot feeds the ledger with the check off too.** `run_fix` files
  its spots through `note_marked` (no picture counted: the critic saw no
  whole picture) and finds the source picture's record by its file name
  (`record_of_picture`) to tell a blind spot. After a Generate the critic
  saw everything, so whatever it did not flag it missed; after a fix it
  saw the spots alone, so it missed only what it had called cleared. The
  same kind marked on the same picture again (Generate Again on a fix) is
  not one more picture with the fault. Not built: switching a pass on from
  the ledger - the hands pass is already on unless unticked, and the
  user's untick is not overridden.
- **The record says what happened.** `record["refinement"]` holds the
  intent, final canonical state, the pass history (each pass with the
  `scores` of its fixes, and `taken_back`), every fault's last score
  (`scores`), what was given up (`left`) and why it stopped;
  `image-studio/visual_critic.log` has each pass's matches, mismatches and
  chosen action. Faces are not matched to characters: a face correction
  redraws every face with every character's description.

### Fix a spot: the user clicks what to redraw

"Fix a spot" under the preview (and on its right-click menu) opens `FixWindow`:
the picture large, each click a square to redraw (the wheel sizes it, a
right-click removes it), what it is (Hand / Face / Something else), how much to
change (Light 0.45 / Medium 0.65 / Strong 0.85) and optional words. Redraw
queues a job with `mode: "fix"`; `Studio.run_fix` composes the picture's own
settings (`fix_base`: no references, pose, face pass or critic), and runs
`face_graph` on the uploaded picture with the squares as crops (`head`
False, so no SAM3 is needed). Only inside the squares changes; the result is
a new history record with `fix`, and Generate Again / New seed retry the fix.
Needs a template with a `face_detail` section: `flux_dev_baseline` and, since
this, `zimage_hq` (which also lets the face pass run on Z-Image).

One-click Find (hands / face / accessories) runs `parts_graph` (SAM3, one
detect per `FIX_FIND` word: an accessory is asked for as glasses, hat,
necklace... since SAM3 has no one concept for it) and `found_spots` squares
each box. A face fix with a SAM3 checkpoint blends back through SAM3's
"face" (`FIX_FACE_MASK`) in the original and the redraw, plus Find's box,
not the oval. Lock mode marks squares (`fix["locks"]`) that `face_graph`
lays back from the original after every crop, so nothing in them changes.

Each fix crop is `FIX_CONTEXT` (1.5) times its spot, redrawn through
`fix_oval.png` (`oval_png(scale=1/1.5, centre=0.5)`), so the model sees the
photo round the spot and only the spot changes. Then ComfyUI's
`StudioMatchTone` (`comfy_nodes/studio_matchtone`, numpy only; copy it into
`custom_nodes` and restart) fits a per-channel curve from the redraw onto
the original over the redrawn part and moves it `FIX_TONE` (0.85) of the
way, so the patch keeps the picture's grade. Without the node the fix runs
and a job note says the colours were not matched.

The fix prompt is only the thing fixed (`FIX_PROMPT`), never the picture's
prompt: glasses redrawn at 0.85 from "a woman, whole figure in view,
dancing" came back as a tiny dancer in the head (2026-09-26). A spot Find
made (not a face with SAM3) is redrawn and blended only in its box grown by
`FIX_AREA_GROW` (`fix_areas` -> crop `area`), not the oval's whole reach.
Inside that box only the thing's own outline is redrawn and blended: Find
keeps the SAM3 word that found each spot (`word`), and `face_graph` asks
SAM3 for it again in the crop, grows it `FIX_SHAPE_GROW` px and multiplies
it by the box (nodes `s0`-`s3`). Live on 2026-09-26 a glasses fix changed
~1,000 px, all on the frames. `found_spots` drops accessories over
`FIX_FIND_MAX` of the picture and boxes mostly inside a kept one of the same
word ("bag" found the apron, so it is no longer asked for).

A spot can be a **freehand outline** (a drag in the window; a click still
makes a square). ComfyUI has no polygon mask node, so `outline_png` draws
the outline as a mask picture at the crop's size, `run_fix` uploads it
(`_outline_masks` -> crop `shape`), and both graphs load it as the noise
and blend mask. **A spot with a photo** (click a marked spot) is not
redrawn by the picture's model: `swap_graph` runs Qwen-Image-Edit 2509 on
qwen_dress.json's loaders, the crop as picture 1 and the photo as picture
2 (`SWAP_PROMPTS`, in the sentence shape Qwen follows), blended back
through the outline, SAM3's word before and after, Find's box or the
oval, in that order. It needs no FLUX, so it works on Z-Image pictures;
its colours are deliberately not tone-matched. Photo spots run first; the
other spots are then redrawn on that picture (loaded by its output name).
The Change strength does not apply to a swap (Qwen draws at denoise 1).

**A fix can end with a face swap** (the window's Face swap row: an
identity, `fix["face_swap"]` its id). After the spots are done (photo
swaps, then redraws), `_face_swap_graph` runs one SAM3 finder
(`faces_graph`, `face:8`) over that result and **every** reference picture
of the identity. The picture's biggest face is the one swapped (the
subject; one identity is one person). Each reference is cut to its own
biggest face (`head_square` at `FACE_SWAP_REF_PAD` 1.8, with the hair):
Partner's references are half-body cut-outs whose face is a tenth of the
picture. Qwen 2509 takes three pictures, the crop and two more, so two
references get a slot each ("pictures 2 and 3") and three or more go in
side by side as one picture 2 (`SWAP_SLOTS`, `ps["sheet"]`). The swap is
colour-matched to the crop it replaces with core `ColorTransfer`
(reinhard_lab, strength `fix["tone"]`), then blended back through SAM3's
"face" before and after, and locks are laid back last. A face swap with no
spots marked is a fix on its own. If SAM3 finds no face, a face-only fix
fails and a fix with spots keeps their result with a note. It needs SAM3
and the Qwen files on the backend, like a photo spot.

Live on 2026-09-26 (Partner, dancing in a meadow, a ~50 px face, 19 s). The
first version used only the first reference, which was her in profile. It
came back as a pale stranger with a white halo round the head. All three
references cut to the face gave her glasses, face shape and mouth, in the
picture's own pose. `StudioMatchTone` on a swapped face posterized it
into cyan and green blotches: its per-channel curves are too steep on
smooth skin. Use `ColorTransfer` there, not the curves.

### Add-ons: LoRAs per model

The user asked (2026-09-26) for "add ons for the current image editor. think jellyfin add
ons but for my specific models. if something i have doesn't work with that model it
isnt showed as a lora". So a LoRA's "server version" is the model:

- **The form offers only what fits the chosen model** (`ImageStudio._offered`:
  `enabled`, and `ig.lora_fits` is not False). `lora_fits` asks every family the model
  runs as (`model_families`: its own and any a backend overrides it with), so a LoRA is
  offered if it works wherever the model may run. Add LoRA lists only those, with a
  "Model not set - may not work" submenu for LoRAs of unknown family (hiding those
  could hide one that works) and "Find more for <model>…" into the catalog. Rows
  already on the form that stop fitting when the model changes are **parked**
  (`parked`, id -> strength), said in a faint line under the rows, and come back at
  their strength when a fitting model is chosen again; `collect` never sends them.
  A saved LoRA mix none of whose LoRAs fit is left out of the Preset row.
- **`enabled`** on a LoRA record (default on) is Add-ons' Turn off: kept installed,
  not offered on the form, and not added as Always on. A LoRA named explicitly
  (history, identity, style) still applies, so Generate Again reproduces.
- **`AddonsWindow`** (the LoRA row's "Add-ons…"): a pill per model plus "Other models"
  (LoRAs whose family fits none of the library's models, like Qwen-Image-Edit's, which
  no form model loads), and two tabs. *Installed* is the model's LoRAs (`sorted_for`:
  fitting, then family not set) as cards: preview, category, strength, trigger, which
  checked backends have the file (`where_installed`), a "Made for" menu, Turn off,
  Always on, Page, Uninstall. *Catalog* is CivitAI's `/models` search with
  `types=LORA`, `nsfw=false` and the model's own `baseModel` names
  (`civitai.FAMILY_BASES`, checked against the live API: Z-Image is `ZImageTurbo` +
  `ZImageBase`, FLUX.1 dev `Flux.1 D`), sorted by downloads, rating or date, with a
  query and cursor paging ("More"). CivitAI lists a model under every base any version
  was trained for, so a card (`card`) is the newest version whose base suits the model:
  the Hands LoRA shows its F1D version under FLUX and its ZIB one under Z-Image. A card
  already in the library (`installed_as`: hash, then `modelVersionId` in `source`, then
  filename) says Installed. Install is the importer's `import_link` into the LoRA
  folder of a backend on this PC that has the model ready (else any with a folder),
  then Check connections, since `compose` leaves out a LoRA the backend's last-read
  list lacks.
- **Pictures are PNG because Tk reads no JPEG**, and CivitAI's image CDN serves JPEG
  whatever is asked for (`Accept`, the URL's extension). `to_png` converts a batch in
  one PowerShell run through Windows' own System.Drawing (script by
  `-EncodedCommand`, pairs as JSON on stdin, a `core.procs.spawn` child), resized to
  `THUMB` px; cached in `image-studio/addon-thumbs/`. Installed previews that are not
  PNG or GIF get a PNG copy there (`previews`). Only a picture CivitAI rates PG or
  PG-13 (`nsfwLevel` 1 or 2) is shown (`safe_preview`, asked for at `width=320`);
  without one the card says "no safe preview". Model-level `nsfwLevel` is a bitmask
  of every image and flags nearly everything, so it is not used to hide cards.
- **Uninstall is two clicks and the Recycle Bin**: `recycle` is `SHFileOperationW`
  with `FOF_ALLOWUNDO`, never a hard delete, for every copy in a backend's LoRA folder
  on this PC, then the record goes. The first click names what uses it
  (`users_of`: identities, styles, saved mixes). A LoRA whose file is only on another
  machine cannot be removed from here and is turned off instead: taking just its
  record out would be undone by the next backend scan (`merge_loras`).

Tests: `test_catalog.py` (engine, CivitAI canned; the PowerShell conversion runs for
real on a BMP it writes) and `TestImageStudioTab` (offering, parking, the window).

### Try On: dressing a person from pictures

**Retired from the form on 2026-09-25.** the user did not want the finished
picture redrawn: item pictures now go into the picture itself (*Item pictures
go into the picture itself*, above). The Try On window, its button, the
picture menu's entry and Generate's dress pass are gone. The engine is kept
(`submit_dress`, `run_dress`, `_dress`, the graphs) only so Generate Again
still remakes a Try On record in history. Reuse on one says it has no form to go back to.

Clothes, hair and accessories from pictures, put on a person who is already
drawn. A Try On job is an ordinary `Job` with `settings["mode"] == "dress"` and
the outfit in `settings["outfit"]`; it has a queue row, a history record
(`dress_record`) and Generate Again like any other.

- **The recipe is `comfy_workflows/qwen_dress.json` plus code.** The file holds
  the loaders (Qwen-Image-Edit 2509 fp8, its 7B encoder, VAE, the Lightning
  4-step LoRA) and two model chains: plain (node 6) and with kingroka's Clothes
  Try On LoRA (node 9, `clothes_tryon_qwen-edit-lora.safetensors`, strength 1.5
  as its page says; CivitAI 1940532, SHA-256 `741606…528f`). `built_by` marks
  it as finished in code (`dress_graph`, `dress_head_graph`), so it is not
  offered as a model's workflow and the fill test skips it. 6 steps at CFG 1
  (the author uses 6-8 on 4-step Lightning).
- **The clothes go on as the LoRA was trained**: one picture, every garment in
  a column on white on the left (`ResizeAndPadImage`, `ImageStitch`), the person
  on the right, the whole canvas 1 MP, and its own prompt, word for word ("put
  the clothes on the left onto the person on the right."). The person's half
  is cut back out with `ImageCrop` at exactly the size it went in: **the
  canvas is sampled at the size it is built at, never resampled**. Scaled to
  1 MP in between, the cut landed a few pixels off and left a white edge. At
  2 MP (`tryon_megapixels`) the person drifted left and went soft; 1 is right.
  The LoRA follows the body well: long sleeves over a t-shirt, ripped jeans,
  a trucker jacket's pockets and buttons (5090, 2026-09-25).
- **The LoRA does not do shoes, hats or accessories** (its author says so, and a
  column of glasses and a necklace was ignored). Hair and accessories are
  Qwen's own multi-picture edit: the person as picture 1, what to put on as
  pictures 2 and 3, two accessories to a pass. **The wording decides whether
  anything happens.** Any "keep the face, pose and background the same" or
  "same framing" clause, and the edit was not made, seed after seed. "The
  person in picture 1 now has the hair of the person in picture 2" and "The
  person in picture 1 wears the glasses from picture 2 and the gold necklace
  from picture 3" were made. Say "picture", as `TextEncodeQwenImageEditPlus`
  labels them. Hair as words alone is "Change the person's hair to …".
- **Hair and head accessories are drawn on a head crop.** On a full-length
  picture the accessory pass was a coin toss: no change, the right edit, or a
  head-and-shoulders portrait with a new face, by seed. So the first run
  (`dress_graph` with `find_head`) draws the clothes and the body's
  accessories (a watch, a belt, a bag: anything `HEAD_WORDS` does not name),
  ends in a preview, and SAM3 finds the faces. The second run
  (`dress_head_graph`) crops `head_region` around the largest one, enlarges it
  to 1 MP, draws the hair and head accessories there, and blends back only
  the person (SAM3's "person" before and after, grown `PERSON_GROW`, softened,
  times a soft rectangle). Through the rectangle alone the crop's redrawn
  background showed as a pale box behind the head. A head crop over
  `HEAD_SHARE` of the picture (a portrait) is the whole picture. Without SAM3
  on the backend it is one run on the whole picture, and the record says so.
  The 5090 has had `sam3.1_multiplex_fp16.safetensors` (Comfy-Org, 1.75 GB,
  in `D:\ComfyUI-models\checkpoints`) since 2026-09-25, which the face pass
  there needs too.
- **Measured on the 5090 (832x1216)**: the clothes alone 15 s; clothes, hair,
  glasses and a necklace 72-74 s (a SAM3 load is ~20 s of it); a FLUX picture
  dressed the same way 60 s end to end. Dressing redraws the person at ~1 MP
  and scales back, so a refined picture loses some of its refine; the face
  pass runs after dressing, on the dressed picture. Fidelity is best on a
  picture of a person in plain clothes. On a FLUX picture whose words already
  named the clothes, the result was looser (a tartan for a buffalo check,
  black frames for tortoiseshell).
- The try-on LoRA and SAM3 are on the 5090; the 3090 has the edit model, the
  Lightning LoRA and SAM3 but not the try-on LoRA, so `dress_route` sends a Try
  On with clothes to the 5090 (the primary first either way) and names the
  missing file when nothing can take it.

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

### The app this window serves: `served`

OpenCode is a coding agent - it edits and runs whatever it is pointed at. The user asked
(2026-09-27) to "use the local llm to make edits (with my input for each step)" on
**this app's own code**, so it runs natively and what keeps it in check is that it asks,
not a sandbox. (It used to run in a Docker container that saw one scratch folder; Docker
was never installed here, the tab never ran, and a sandbox cannot edit this repo.)

- **`ServerSpec.launch()`** writes OpenCode's config to `OPENCODE_STATE`
  (`%LOCALAPPDATA%\StudioAssistant\opencode`, never the workspace), writes a fresh
  random password to `server.key` there, and starts `opencode serve --hostname
  127.0.0.1` with `OPENCODE_CONFIG` and `OPENCODE_SERVER_PASSWORD` in its environment,
  in the workspace (`OPENCODE_WORKSPACE`, default this repo), through
  `core.procs.spawn` - so it ends with the window, like a bridge. `opencode_exe()`
  starts npm's native `opencode.exe` directly, not the `.cmd` shim. Install is `npm
  install -g opencode-ai`; the Start button says so when it is missing.
- **The password is not optional.** A coding agent's HTTP API on loopback is reachable
  by any web page in a browser here; OpenCode answers 401 without HTTP Basic
  `opencode:<key>`. The bridge reads the key file on every request, so a restart (new
  key) needs nothing from it. A 401 says who restarts the server.
- **`OPENCODE_PERMISSIONS`** in `opencode_config()`: read, search, list, todo, skills,
  subagents and OpenCode's own questions are `allow`; `edit`, `bash`, `webfetch`,
  `websearch` and `doom_loop` are `ask`; `external_directory` is `deny`. A key left
  out falls back to OpenCode's default, which is allow - `test_opencode_config_asks_
  before_every_change` holds the set. Each add-on MCP server adds `<name>_*: ask`,
  because OpenCode runs MCP tools without asking otherwise; add-ons only add
  permissions, never loosen one.
- **The model gets the loaded window.** `context_window()` is passed as the model's
  `limit.context`, or OpenCode never compacts and overruns a 32k load.
- **On this repo OpenCode reads `docs/OPENCODE.md`, never this file.** OpenCode puts
  the workspace's AGENTS.md whole into every request; this one is ~60k tokens, more
  than the loaded window, so the task and everything it read were compacted away and
  it "forgot" what it was doing. When the workspace is this repo, `launch()` sets
  `OPENCODE_DISABLE_PROJECT_CONFIG=1` (stops the root AGENTS.md) and the config's
  `instructions` names the brief (`OPENCODE_BRIEF`). Keep the brief short (a test
  holds it under 8k chars); put new must-know rules there, detail here. Another
  workspace keeps its own AGENTS.md. `OPENCODE_PROMPT` must not tell the model to have
  OpenCode read AGENTS.md.
- **The tab codes on the best coder the host has.** `ServerSpec.model_for` takes
  `best_coder(ids)`: names carrying a `CODER_HINTS` word, no bigger than `CODER_MAX_B`
  (40B, the LLM PC's 24 GB), ranked dense before MoE (an `-aNb` MoE counts half its
  total) then by size. No coder on the host: the shared model, silently.
  `STUDIO_MODEL_OPENCODE` still pins one. `fit_window()` at launch loads that model
  with at least `OPENCODE_CONTEXT` (64k, capped at its maximum), reloading only it.
- **A follow-up continues the last session.** `opencode_ask` without `session_id`
  reuses the one in `OPENCODE_STATE/last_session` (a local model often drops the id,
  and a fresh session knows nothing); `new_session: true` starts over.
- **Under 64k of window it warns.** `context_note()` names LM Studio's Context Length
  in `opencode_status` and after each ask. (Restart OpenCode is only in the Add-ons window; the note says to reopen the app.)

**How a step reaches the user.** `opencode_ask` sends the task with `prompt_async`
and `run()` follows the session: each pass lists `/permission` and `/question`,
keeps those of this session or a subagent's (`Family`, by `parentID`), and puts each
to the user with `core.mcp.elicit()` - the MCP client's user, not the model. The
reply goes to `/permission/{id}/reply` (`once` | `reject`, with the note as `message`,
which OpenCode's model reads as the user's feedback - never `always`, see *grants*
below) or
`/question/{id}/reply`. The loop ends when `/session/status` has the session idle
twice (`SETTLE` covers a task not yet marked busy); the result lists the user's
decisions, then each assistant message's text, tools and files. Time the user spends
deciding does not count against the call's `timeout`; past it, `opencode_wait`
picks the same session up. There is no tool that approves anything, and
`opencode_put_file` is gone - it wrote on the model's word alone.

- **Cancel is a Stop.** An elicitation answered `cancel` - the card's Stop, the send
  button's Stop, a closed tab - aborts the session. `decline` is a refusal. A client
  that did not offer elicitation (a piped CLI, another MCP client) gets every step
  refused and the session stopped: nothing changes on the model's word.
- **Routes** live in `ROUTES`, checked against OpenCode's `/doc` by `opencode_status`
  as before. Read off OpenCode 1.18.32; the shapes the tests fake are the ones the live
  server returned (`permission`, `patterns`, `metadata.filepath`/`diff`/`command`,
  `always`).
- **Attachments** from outside the folder are copied into `.studio-attachments/`
  in it (git-ignored); one inside is named by its path there.

- **Delegate changes promptly.** The outer tab sends the user's request and
  constraints to `opencode_ask`; OpenCode reads the project rules and locates the
  implementation. It need not discover every function before handing off. After
  three workspace reads without a handoff/session inspection in the current run,
  the shared executor removes exploration tools from the request and refuses them
  at dispatch until a successful handoff/session inspection. It can still answer,
  ask a focused question, recall evidence, or delegate a review explicitly without
  edits. The guard also covers calls hidden inside made tools; it never approves
  an OpenCode permission request.
- **Workspace reads fit the executor's result budget.** `opencode_read_file`
  returns at most 6000 characters plus a header with the next `start` character
  offset. `opencode_search_files` does bounded case-insensitive search - literal by
  default, or a regex with `regex=true` - returning line numbers and offsets for
  that reader. Search and listing report
  partial results when bounded; normal listings/searches skip `.runtime`, `.work` and
  `.studio-attachments` along with dependencies. Explicit paths still work.
- **Stopping is not evidence of an edit.** A read-only journal says no
  edit-capable tools ran. If any edit-capable operation was attempted, the stop
  message says the project may have changed, including failed/unknown outcomes.

Live-checked (2026-09-27, scratch project, qwen3-coder-30b): the edit's diff was
asked and allowed, `python hello.py` was asked and refused with a note, OpenCode's
reply quoted the note, the file changed, the server ended with `stop()`; an add-on
MCP server (`npx -y @modelcontextprotocol/server-sequential-thinking`) connected and
its tool was asked before it ran.

### OpenCode's tasks: copies, checkpoints, tests, grants

Added 2026-09-27 (bridge 3.0) because OpenCode edited the live checkout the app runs
from, nothing could be taken back, nothing ran the tests, and "always allow" was
OpenCode's to keep and invisible.

- **Each task has its own copy.** When the workspace is the top of a git repository
  with a commit, `start_task()` makes a git worktree under `OPENCODE_STATE/worktrees/`
  on branch `opencode/<stamp>` from `HEAD`, and creates the session *in that folder*:
  every session route takes `?directory=`, and OpenCode runs one instance per folder
  (live-checked on 1.18.32: the session reported the worktree as its directory, and
  `/permission`, `/session/status` and `/event` are per folder). `task_dir(sid)` is
  that folder; `messages`, `prompt`, `abort`, `settle`, `follow` all pass it. The
  record is `OPENCODE_STATE/tasks.json`. The copy starts from the last commit - the
  user's uncommitted edits are not in it. A folder that is not a repo gets no copy and
  works in place, as before (`isolated: false`).
- **After each ask that ends idle** (`after_ask`): the changed files' tests run in the
  copy - `tests_for` maps `core/x.py` to `tests/test_x.py`, `apps/<app>/mcp.py` /
  `ui.py` to `tests/test_<app>.py` (`TEST_ALIASES` for comfy, imagegen), a changed test file to itself - with `TEST_TIMEOUT`; then a
  checkpoint commit (identity `OpenCode <opencode@localhost>`) on the task's branch.
  The report says passed/FAILED with the output's tail. These tests run code the user
  approved edit by edit, in the copy, without another ask.
- **Merge, undo, discard are the user's.** `opencode_merge` squashes the branch into
  The user's current branch as one commit, after an elicitation showing the diff;
  it refuses if the user has uncommitted edits in the same files, and a conflict is
  `git reset --merge` and a sentence. `opencode_undo` resets the copy to the previous
  checkpoint (and the next prompt tells OpenCode so), or for a merged task `git
  revert`s the merge commit. `opencode_discard` removes the copy and branch. Each asks
  through `confirm()`; no client to ask means no. The tool names avoid
  `approve|allow|...` (`test_nothing_the_model_can_call_approves_or_writes`).
- **Grants are the bridge's.** "Always allow" is stored on the task (`add_grant`,
  OpenCode's `always` patterns) and OpenCode is answered `once` every time, so a grant
  is visible (`opencode_grants`, `opencode_status`), revocable (`opencode_revoke`),
  survives a server restart and ends with the task. `granted()` matches with
  `fnmatch`, where `cmd *` also covers `cmd` alone. The card says "... for this task".
- **Events, not a 1-second poll.** `Events` reads `/event` (SSE) on a thread and wakes
  `follow()` on any event but the connect/heartbeat; with it up the loop looks every
  `EVENT_WAIT` s, and it falls back to `POLL` when the stream fails or drops. The
  stream never decides anything - the status and permission lists still do.
- **Context is reported.** `context_report` adds to each report the last request's
  tokens (`info.tokens`: input+output+cache) against the loaded window, warns at
  `CONTEXT_HIGH` %, and counts compactions (a `summary` message or `compaction` part).
- **Direct mode** (the tab's header "Direct" button, `Chat._direct_turn`): the user's
  message is `opencode_ask`'s prompt as typed and the report is the reply - no model
  briefs OpenCode, so one context on the GPU, not two. Approvals come up the same way.

### OpenCode's Add-ons

The **Add-ons** header button on the OpenCode tab: the user asked for "plugins like the
image creator has, but for a coding agent". `AddonsWindow` mirrors the LoRA Add-ons: a
pill per kind, Installed and Catalog tabs, Turn off, Remove on a second click.

- **Kinds are OpenCode's own**: `mcp` (config `mcp`), `plugin` (npm packages, config
  `plugin`), `skill` (SKILL.md folders, config `skills.paths`). Records are in
  `OPENCODE_STATE/addons.json`; `config()` adds the enabled ones to the opencode.json
  written at each start, so the window offers **Restart OpenCode** after a change. It
  refuses while the tab is working.
- **Catalogs**: the official MCP Registry (`/v0/servers`; only `isLatest` entries;
  a way to run one is an npm or PyPI stdio package - `npx -y pkg@ver` / `uvx
  pkg==ver`, which OpenCode starts fine on Windows - or a streamable-HTTP/SSE remote;
  container images, .NET tools and packages needing a positional argument are not
  offered), npm search `keywords:opencode-plugin`, and a GitHub repo's tree
  (`anthropics/skills` by default; two API calls, then SKILL.md front matter from
  raw.githubusercontent.com per card). A server's required environment variables and
  header placeholders become fields before it is filed; secret ones are masked and
  kept in the state folder, as OpenCode's config keeps them.
- **Installed MCP servers say what OpenCode made of them**: `live_status()` reads
  `GET /mcp` (connected / failed and why / not loaded yet).
- **Remove**: a downloaded skill's folder goes to the Recycle Bin
  (`apps.image_studio.addons.catalog.recycle`); a folder the user pointed at is left alone.
- Threads post back with the window-level `("call", None, fn)` event, which `_handle`
  runs before any tab lookup; a stale answer is dropped by `gen`.

### OpenCode's trainer: Claude Code, from outside the app

The user asked (2026-09-28) for Claude "beside [OpenCode] as a trainer", and then, firmly,
**not through the API**: he has a Claude plan and will not buy API credits. So the
trainer is Claude Code itself, in the Claude app, and this repo gives it an MCP server,
`apps/opencode/trainer_mcp.py`, registered in `.mcp.json` as `opencode-trainer`. Sitter
says "review the last OpenCode task" or "teach it X" there; Claude reads through the
server and keeps or forgets lessons. The app calls no model for this, and **nothing
here may call Anthropic's API** - a first version did (a Trainer menu, a key window),
and was thrown out for that reason. The community plugins that put a Claude
subscription inside OpenCode are against Anthropic's terms; do not wire one in.

- **It reads what the window already saves**: the OpenCode tab's task records
  (`tasks/opencode/*.json` beside the settings - the whole conversation, OpenCode's
  reports and their "refused: ..." lines), `tasks.json` for what a session changed
  (its merge commit, or the diff in its worktree), and OpenCode's own session when the
  server is up. `trainer_tasks` flags trouble per task (`trouble_in`: an unfinished
  status, a correction, undo/discard, refused steps, failing calls).
- **It teaches only through the notebook** (`trainer_keep` / `trainer_forget`), with
  the source `trainer`, which outranks everything but the user's own lessons
  (`lessons.PRIORITY`). It never forgets a user lesson unless told `user_agreed`.
  Each change rewrites `lessons.md` in OpenCode's state folder, so OpenCode reads it
  on its next prompt.
- **Two processes, one file.** A `Notebook` re-reads its file when it changed on disk
  since it last read or wrote it (`_sync`, on find/add/remove/ordered). Without it
  the window, holding its old list, saved over the trainer's lesson on its next add,
  and never saw the lesson in `fresh()`. With it, a trainer lesson reaches the tab's
  next request without a restart.

### The phone: chat and pictures as a web page

The user asked (2026-09-29) for "a way for me to use the llm as chat and image gen on my
phone". It is `apps/phone/server.py` and one page, `apps/phone/page.html`, started by
`Studio Assist Phone.cmd`: a stdlib `ThreadingHTTPServer` on this PC, port 8765. It
is its own process, not part of the window, and needs the window neither open nor
closed.

- **The phone is a browser and nothing else.** No app to install but Tailscale. The
  conversation is kept in the phone's `localStorage`; the server keeps no chat.
  "Add to Home Screen" makes it look like an app (`/manifest.webmanifest`, and
  `/icon.png` drawn by `make_icon.render`).
- **It is the desktop's engine with no window.** `Phone.chat` streams through
  `eng.LLM.stream` after `eng.fit_model(..., keep=<models mid-reply>)`, so a model
  that is not loaded goes onto an empty card with a window that fits the
  conversation, and one already loaded and big enough is left alone: nothing is
  unloaded for a reply that needs no load. `Phone.generate` is `Studio.submit` on the
  real library, so routing, the face pass, identities (FaceFusion) and History are the
  Image Studio's, and a picture made on the phone is in the tab's History.
  `Studio.make_room` is `Phone.make_room`: LM Studio's models off the shared card,
  but for one a reply is streaming from.
- **What the phone server cannot know.** Whether a desktop tab is mid-request. A load
  it makes beside one cuts that request off, as any load does (see *The window the
  model is loaded with*); the desktop reloads its own model on its next turn.
- **The hands pass is asked for on the phone, not assumed.** The form runs it on every
  Generate. Live on the 3090 a picture of a fox took 348 s, the first minute of it the
  picture and the rest the hands pass. `hands: true` from the page's "Fix hands" turns
  it on.
- **A picture on the 3090 costs the next reply a model load.** The picture clears LM
  Studio off the shared card, and the next message loads the model again: 110 s for
  the 30B, measured live, with "Loading ..." shown on the phone. A picture routed to
  the 5090 costs chat nothing.
- **Chat has no tools.** Its prompt (`CHAT_PROMPT`) says so, for the reason the Chat
  tab's does: a model that cannot reach an app must not say it changed one. The page's
  `system` messages are dropped (`clean_messages`); the prompt is the server's.
- **Who is served** (`Access`). This PC and any tailnet address (100.64.0.0/10: the
  user's own signed-in devices, already authenticated by WireGuard) with no question.
  By default the server listens only on 127.0.0.1 and this PC's tailnet address, so
  the home network cannot reach it at all. `--lan` listens everywhere, and a private
  address must then give a six-digit passcode once (kept in
  `%APPDATA%\StudioAssistant\phone\phone.json`; the cookie's hash is kept, not the
  cookie; five wrong guesses lock that address out for ten minutes).
- **A page in the phone's browser must not be able to use it.** A request is refused
  when its `Host` is a name that is not this PC's (DNS rebinding; an IP address is
  always taken), when its `Origin` is another site, and when a POST is not JSON (a
  form cannot send that without asking first).
- **Pictures are named by record, never by path.** `/picture/<record id>/<n>` reads
  the History record and serves its file only if it is inside the History folder.
  Thumbnails are JPEGs made by one PowerShell run (`make_thumbs`) into
  `phone\thumbs`, not the tab's `.thumb.png` files beside the pictures, which are
  96 px and the tab's own.
- **HTTP/1.0 on purpose.** Every answer ends with the connection, so the streamed
  reply (lines of JSON: `{"note"}`, `{"t"}`, then `{"done"}` or `{"error"}`) needs no
  chunk framing, and a phone that walks away is seen at the next write, which ends the
  request to LM Studio too.
- Windows Firewall asks once whether Python may accept connections; without "Allow"
  the phone gets no answer. The server does not touch the firewall.

### The Nodes view: a picture's graph in ComfyUI's own editor

The user asked (2026-09-27) to see the pipeline's nodes and edit specifics "from the app",
like the Milanote tab, and then for it to live in the ComfyUI tab rather than a tab of
its own. The ComfyUI tab has a **Chat | Nodes** switch above its transcript
(`apps.comfyui.nodes_ui.NodesView`, built only for the `comfyui` tab). Nodes puts ComfyUI's
page where the transcript is, with a backend picker (the Image Studio's backends plus
the tab's own `COMFYUI_URL`, shown first), a step picker and Reload. Chat puts the
transcript back. The conversation and its bridge are untouched either way.

- **While Nodes shows, the tab counts as a window.** `Chat._holds_window` (a panel tab,
  or `nodes_view.on`) is what `_select` and `_apply_status` ask. It hides the composer,
  focuses the window, and disables New chat and History, as on the Milanote tab.
- **The window is the session's `browser`.** `_close_tab` releases it before the frame
  goes and `Session.close` ends it, with no code of its own. Its workers come back as
  `("nodes", sid, callable)`, which `_handle` runs on the UI thread.
- **The Image Studio's Nodes button** (2026-09-28: floats over the picture's top-right
  corner rather than sitting in the action row - Fix a spot took its old seat there;
  Show nodes is still on the picture's right-click menu too) calls
  `Chat.open_nodes(steps, url, name)`. That opens or selects the ComfyUI tab, switches
  it to Nodes, and loads the picture's Picture step on the backend that made it. The
  picture's steps sit on the bar until another backend is picked by hand (`_switch`),
  because they were made on the first one. The first version took over the Image
  Studio's own body instead. It moved here because two windows cannot share one
  profile.
- **A job still waiting has no graph yet** (`Job.graph` is set inside `run_job`, once
  its lane starts it - not at enqueue) **and neither does the form before Generate is
  pressed.** `ImageStudio._selected_steps` falls back for a queued job, and
  `ImageStudio._form_steps` builds one for the form itself, both through
  `ig.preview_graph(plan)`: `compose()`'s `Plan` (already no I/O, used for the form's
  warnings) filled with `fill()` - the same adapter `run_job` uses - but with each
  reference's local file *name* standing in for an uploaded one, since nothing is sent
  to the backend. `_show_nodes` tries the selection first, the form second, so Nodes
  is connected to whichever backend would actually run it (Auto included) and never
  blocks: nothing opens or loads until it is pressed.

`apps/comfyui/view.py` holds the window and the graphs:

- **The window is the Milanote one.** `ComfyBrowser` subclasses `apps.milanote.milanote.Browser`
  (`name` for its sentences, `target()` for which page). It has its own profile,
  `%LOCALAPPDATA%\StudioAssistant\comfyui-browser` (`STUDIO_COMFY_PROFILE`). Embedding,
  clipping the caption with `measure()`/`fit()`, and `release()` before the frame goes
  all work as *The tab that holds a window* says. A picture made on the other backend
  navigates the same window there.
- **A graph goes in through `app.loadApiJson(graph, name)`** over DevTools. It takes the
  API-format graph exactly as queued and opens it as a workflow tab of its own, the
  nodes in columns by their links. Loading queues nothing. Run in the page is the
  user's, and what it makes goes to ComfyUI's output folder, not History. The bar says
  so.
- **Wait for the old session, not just for `app`.** The page reopens its last
  session's workflow tabs about 0.2 s after `app.vueAppReady`, and that restore drew
  over a graph loaded in between. We saw 10 nodes where 18 were loaded. `READY` also
  waits for the title to name a workflow. `show()` compares the node count with the
  graph's and loads once more if they differ.
- **Fit the view by bounds, not by the Fit View command.** A loaded graph lands
  wherever the view was, often on empty canvas. `Comfy.Canvas.FitView` animates by
  frames, which a window out of sight never draws, so it did nothing. `LOAD` calls
  `ds.fitToBounds` on the nodes' box, with 12% extra on the left for ComfyUI's toolbar.
- **Leaving a page answers "Leave app?".** A loaded graph is an unsaved workflow, so
  going to the other backend, or Reload, raises ComfyUI's beforeunload prompt, and
  unanswered the navigation hung ("Opening ComfyUI…" for good). `leave()` sends
  `Page.navigate`/`Page.reload` itself and accepts `Page.javascriptDialogOpening`.
  Nothing is lost: ComfyUI keeps each open workflow as a draft in the profile and
  reopens it on that page's next visit.
- **A record's steps are its graphs** (`graph_steps`), in the order they ran: Try On's
  garments, then Picture (`graph`), Face pass, Real-face paste, then `passes`.
  `passes` is new: `_run_pass` appends each `{"label", "graph"}` to `Job.passes`, and
  `record_for` saves them. That covers eyes, hands, glasses, Fix a spot's spots and the
  Visual Critic's redraws. Records from before this change have no `passes`, only
  their other steps. Every PNG ComfyUI saved also carries its own graph in a `prompt`
  text chunk. Dragging a picture onto any ComfyUI page opens that last step.

### The Scene Builder: a stage for the Image Studio

**Scene Builder…**, under the Image Studio's prompt field, opens one window laid out like a
slicer: library and object list on the left, the viewport with Move / Rotate / Scale in
the middle, the selected object's controls on the right, File / Build from photo /
Suggest details along the top, and Model / Generate picture along the bottom. Undo and
History have their own row below the editing tools. People have Object / Pose / Look
inspector sections; clicking a body part opens Pose. The scene row (no object selected)
has Scene / Camera sections the same way - Scene details, Enrich and what the picture
follows on one, Frame and the camera's lens/orbit/pitch/distance/aim on the other -
reselecting the row resets to Scene, as reselecting a person resets to Object. Image
Studio has Image / People /
References / Settings sections, keeps Generate outside the scrolling form, and puts
LoRAs inside Advanced in Settings. Section changes retain settings and reset scrolling.
Generate inside the builder includes its arrangement; a plain Generate in the form does
not. It blocks out a picture; it is not a 3D package. `apps/image_studio/scene/scene.py` is the engine
(no tkinter, tested headless), `apps.image_studio.scene.ui.SceneBuilder` the window, a collaborator
of `ImageStudio` exactly as `CharacterCreator` is. The rules:

- **Generate goes through the Image Studio, never beside it.** The builder draws the
  scene's maps (`scene_maps`, under `image-studio/scenes/renders/`, named by content
  hash), writes the scene's words into the form's Scene field, and calls
  `ImageStudio.generate(extra=...)`. `extra` lays the frame's size, the maps as
  `references` (the form's pose, composition and source slots replaced; its face and
  style kept), their strengths, the denoise when the frame is one of them,
  `scene_layout` (the whole scene) and `scene_file` over the form's settings for that
  job alone - the form's own slots are not touched: History records them, and the next plain
  Generate from the form carries none of them. Routing, refusals, the queue, the face
  pass and Generate Again are the Image Studio's, unchanged.
- **The scene is sent as what it means, not as the grey frame.** Image to image
  from the frame (until 2026-09-25, at denoise 0.7) copied the mannequins' blocky
  look into the people: any denoise low enough to keep the layout keeps the
  shapes too. For a model whose workflow has the ControlNet inputs (the FLUX
  baseline) the builder sends two maps instead, each with its own slider (0 is
  off), and the words say how things look:
  - **Pose** (`pose_png`, reference kind `pose`, default 0.85): every person
    and crowd member's skeleton (`rigs`: the same placement `painted_pieces`
    gives their faces) projected through the camera as OpenPose, drawn by
    `apps.image_studio.scene.pose.render_figures`, far to near. Head points are dropped as
    DWPose would miss them - nose and eyes on the side facing the camera, the
    far ear in profile - and the 68 face dots are `apps.image_studio.scene.pose.FACE` turned
    with the head in 3D (`FACE_UNIT` is half the eye gap), drawn whenever the
    nose is seen, profile included, or the person comes back seen from behind.
    A joint with something more than `HIDDEN_BEHIND` (0.3 m) nearer at its pixel
    is dropped, as a photo hides it (the depth map's z-buffer): live on the 5090,
    crowd limbs drawn through the man in front turned him round; hidden, he faced
    the camera on the same seed.
  - **Layout (depth)** (`depth_png`, kind `composition`, default 0.55): a
    z-buffer of 1/z over every face, the floor and the inward walls (`_fill_depth`:
    1/z is linear across a flat face on screen), grey from farthest (black) to
    nearest (white), sky black - Depth Anything's convention, which Union Pro 2.0
    learnt. 512 px on the long edge; the ControlNet scales it. Each body (person,
    crowd member, prop) has its own depth stretched `DEPTH_RELIEF` (3x) about its
    middle, where it is on screen unchanged, and the grey spans `DEPTH_CLIP`'s
    percentiles, not min to max: in true 1/z a person at 5 m is a flat cut-out,
    where Depth Anything's maps give bodies rounded relief.
  - **Grey frame kept** (`frame_keep`, kind `source`, default 0 = not sent): 1 -
    denoise. 0.1-0.25 pins props and exact framing on top of the maps.
  A model with neither ControlNet input gets the frame alone, at
  `FALLBACK_KEEP` 0.3 kept at least (the old 0.7 denoise) - its shapes become the
  picture's, so a lumpy mannequin was drawn as two people stacked. Z-Image takes
  both maps since 2026-09-29 through Alibaba PAI's Z-Image-Turbo Fun ControlNet
  Union 2.1 (2602, 8 steps), a *model patch* (`model_patches`, `ModelPatchLoader` +
  `ZImageFunControlnet`): it patches the model, not the conditioning, so in
  `zimage_hq` only the first pass samples with it (node 56); the face and refine
  passes keep node 4 unpatched. `SceneBuilder.takes()`
  reads which of the three a model's workflows have; `check()` refuses only a model
  with none. A backend lacking the ControlNet file is compose's warning, and that
  picture is made from the words. A scene saved with `redraw` opens with the
  defaults. The maps are named in the status line after Generate, for looking at.
  Measured 2026-09-25 (FLUX on the 5090, seed 4242, ~16 s): the same Oktoberfest
  scene from the frame at 0.7 came back a flat vector illustration; from the maps,
  a photograph with the two people where and as they stand. The crowd's raised
  arms were not kept at 0.85/0.65: small figures follow the pose loosely.
- **The viewport is the frame.** The canvas always looks through the one camera; the lit
  rectangle is `render()` at the generation size, the same polygon list `png()`
  rasterises, so what is inside it is exactly the reference. Outside it is dimmed
  context. The metre grid and the selection outline are the window's only: the picture
  must not be told the floor is tiled. The rectangle is labelled the **viewfinder**
  with the camera body, lens and size, and it is also what is described: an object
  outside it is left out of the words (below), and so are walls or a floor it does not
  show (`room_seen`), with a note saying so.
- **The camera body sets the frame and is named in the form.** A camera profile has a
  `format` ("1:1" or "3:2", `imagegen.CAMERA_FORMATS`; starter cameras saved before it
  take `DEFAULT_CAMERA_FORMATS` by id). Choosing it bakes `profile`, `body` (its name),
  `format` and `chemistry` into `scene["camera"]` and moves the frame to one that
  camera shoots (`camera_frame`: kept upright if it was; `FORMAT_FRAMES`, the 3:2 frames
  are 1216 x 832 / 832 x 1216); the Frame choice then offers only those, and
  `clean_scene` holds a saved scene to it. `camera_words` says "Shot on a <body>" when
  the chemistry does not already name it.
- **Shot on is a deck of cards above the Scene field.** The Image Studio form shows
  one camera card at a time (its picture, else its name), flipped with ‹ › or the wheel
  over it; the card showing is the choice, `settings["camera_profile"]`
  (`_build_camera_deck`, `set_camera`). A plain Generate gets its words
  (`imagegen.camera_profile_words`, after the Camera field) and its shape
  (`camera_size`: the model's size reshaped to 1:1 or 3:2 at about the same area,
  held the same way up; a size typed in Advanced wins). A scene job skips both - its
  words carry the camera and its frame is sent. With the Scene Builder open the deck
  and the builder are one choice: a flip calls `_set_camera_profile`, and the builder's
  `_words` calls `ImageStudio.show_shot_on`, which turns the deck to the scene's camera.
  A new scene (and Reset camera) starts with it through `_bake_camera`.
- **Descriptions are sent as written.** `scene_text()` adds only what the words cannot
  know - where each object is in the frame, which way a person faces, a non-standing
  pose - in parentheses before the user's text (and a person's look between the two),
  and it says "a person" so the anatomy constants apply. An object outside the frame is left out of the words and said so,
  and so is an object with no description. A test holds punctuation and case verbatim.
  "Outside" is none of its box on screen: its middle alone dropped a person framed head
  and shoulders, whose middle is below the frame; left / centre / right is the seen part's.
- **A person is said from their controls, as the pose map draws them.** Each person's
  line is `Name (a person, where, facing, [gaze], framing, [named pose]): look.
  Posture. Description`. `posture_words` reads the posed skeleton, not the sliders, so
  a combination says what it looks like: the torso's bend, lean and turn; each arm
  from where its wrist ends up against the crown (`HEAD_TOP` above the head joint),
  shoulder and hips ("raised above the head", "reaching forward at shoulder height",
  "bent, the hand in front of the chest", "swinging forward"...), both arms in one
  phrase when they match (`BOTH_ARMS`); the legs (stride, weight on one leg, wide
  stance) unless a named pose in `LEG_POSES` says them; the head's nod and tilt.
  **Look at** (the L tool): a click on a person opens a ring round their head of
  `HEAD_POSES` - ahead and the eight directions, left and right as the camera sees
  them (`head_pose` flips the sign for someone facing away) - with Camera and
  Point... under it. Point makes the next click the point (a face at its depth,
  else the floor, else 30 m out along the ray). The point is
  kept as `look_at` and `aim_head` solves Turn and Look down to it - again on every
  `changed()` and drag, so the head follows the person. The head only, clamped to its
  sliders; moving a head slider by hand drops the point. A crowd has none.
  Left and right are theirs, as captions say them. `gaze_words` says where the head
  looks when that is not the body's way ("head turned towards the camera"), and
  `framing_words` how much of them the frame shows ("seen from the knees up"). A
  look with a Gaze keeps it: the head words are left out rather than contradict it.
  Heights, degrees and body words are not added: the look's sliders already say
  build and height, the anatomy constants say natural proportions, and numbers do
  little in a prompt. Live (2026-09-25, same seed): an arm raised in both the map and
  the words was drawn raised; a front-on carrying pose (forearms towards the
  camera, so short in the map) was not, in one seed of two.
- **Each person carries their own look.** A person object has `look` (the Image
  Studio's `LOOKS` slots and `SLIDERS`, sparse, cleaned by `clean_look`) and
  `character`. The inspector's Look section is the form's own `look_rows` /
  `slider_rows`, one section at a time. Choosing a character copies its look
  (`character_look`: blank where it has none, the expression and gaze kept) and, if the
  person still has the default name, its name; it is a copy, like the form's. The look is
  said in that person's line, `Name (a person, where, facing): look. Description`.
- **Chest size can drive a LoRA.** Library records assign `body_control` to
  `chest_female` (signed strength, scaled by the -3..3 slider) or `chest_male`
  (fixed strength with the documented male size phrases). `compose` requires
  a known matching family and an installed file, and reports when only words
  are available. Scene Builder reads the person's look from `scene_layout`,
  since generation deliberately blanks the form's person. Multiple people
  cannot receive independent settings from a whole-image LoRA; report that
  limitation instead of applying one person's size to everyone. The installer
  `tools/install_chest_loras.py --folder <local LoRA folder>` downloads and
  hash-checks CivitAI versions 2520278 (female, Z-Image Turbo) and 3155339
  (male, Z-Image Base); the latter still needs visual verification on Turbo.
- **The mannequin wears the look, because the maps are drawn from it.** The depth
  map carries each body's outline (and the frame, when kept, its colours): a
  heavyset person drawn as the rest mannequin is pulled thin again, so `painted_pieces` builds each person to `body_shape(look)` - the Weight, Muscle and
  Chest size and Height sliders, plus a Body type word `BUILDS` knows, as steps added to them (clamped,
  so "obese" does not push Weight past +3) - and dresses them in `outfit(look)`: the
  Clothes slots colour the body's regions they cover (a t-shirt the upper arm, a
  sweater the forearm too), a dress, skirt or long coat adds a hem that follows the
  legs. The Shoes slot is read for a kind (`SHOES`): heels tip the foot onto its toes
  over a heel post, sneakers get a thick light sole, sandals a sole and a strap under
  a bare foot, boots a sole and a shaft that replaces the shin's lower part (a tube over
  it sorts behind the shin's long faces and vanishes; under trousers it is left out),
  anything else a thin dark sole. A sole lifts the person, as it does. The Accessories slot puts a hat (`HATS` names it, `HAT_SHAPES`
  builds it: a cap with its visor, a beanie, a brimmed hat, a top hat, a hard hat -
  site yellow unless a colour is said) and glasses, sunglasses or goggles on the head,
  as part `head`, so a click on them poses the head. The Hair section gives hair
  (`hairdo`): a scalp lofted over the skull from a `hairline` (the head's faces under
  it are dropped, or a big one sorts in front and shows through), a fall down the back
  as long as the style says, a bun, ponytail or braids; under a hat only what hangs
  below it. The colour is the first `CLOTH` word in the garment ("black
  leather jacket" is black), else the slot's default. Costume words are drawn too: a
  dirndl is a dress to mid-calf with white puffed sleeves and an apron (each piece's
  colour from the words next to it, `_near`: "a green dirndl with a pink apron"),
  lederhosen are knee-length with braces, a flower crown (`HATS` "crown") is a ring of
  leaves and flowers of every colour unless one is said, and an alpine / German hat has
  a band and a feather. `HELD` puts carried things in the Accessories slot on the
  mannequin: an accordion across the chest (the Carrying pose puts the hands on it)
  and a beer stein upright in front of the right palm, or both when the words say
  more than one. Anything else typed in a slot is still sent as written; it only
  goes undrawn. What is not worn is the object's
  colour. The words are still sent as written; this only draws them. Every look edit
  in the inspector goes through `changed()`, so the viewport follows each keystroke
  and slider step.
- **The thumb is on the outer side of the hand.** `hand_joints` is built with the
  palm facing +z, the front at rest, so the thumb and index finger are toward
  `sign` (+x on the left hand, -x on the right), away from the body. Until
  2026-09-26 they were at `-sign`, a mirrored hand with the thumb by the thigh,
  and the pose map sent DWPose that mirrored hand too. The mesh and the map
  both come from `hand_joints`, so a fix there fixes both
  (`test_the_thumbs_are_on_the_outer_side_of_forward_facing_palms`).
- **With people in the scene, the form's person is blanked for the job.** `generation()`
  lays empty slots, zero sliders, no `character` and no `item_refs` over the form, or the
  picture gets the form's person as well (an extra person, or two blended). The first
  person added takes the form's look and character so nothing is lost. A scene
  character's identity is added to the form's ticked ones for the job (`scene_identities`,
  popped by the window). Item pictures are not sent from a scene: `compose()` matches them
  to the form's clothes, which are blank, and no workflow takes one yet anyway. A scene
  of props alone leaves the form's person alone.
- **Everything stands on its floor.** An object's lowest point is put at its position's
  y (`object_pieces`), so a crouch drops the hips, a kneel puts the knee down and a
  tipped drum lies on the floor; y is the floor it stands on (a platform, a step).
- **The floor can be shaped** (2026-09-26, the user: "an S curve to shape the floor",
  then "a grid of dots 4x4 ... add in presets like a hill"). `room["grid"]` is 4 x 4
  handle heights, rows back (-z) to front, columns left to right, spread over the
  room's width x depth round the origin - walls or not, so resizing the room
  stretches the shape and toggling walls does not change it. **Floor shape** under
  Floor and walls is the floor from above, shaded by height: drag a dot up or down,
  right-click it back to 0, "Shape like…" for a preset (`FLOOR_PRESETS`: hill, dip,
  ridge, rise behind, bank left / right, bowl, rolling), Flatten; without walls it
  also has the area's width and depth. `Ground` is the surface: a bicubic Bezier
  patch of the 16 handles, each parameter run through smoothstep first. So a dot
  pulls a broad area (the ask: not "one vertex only"), the surface stays inside the
  handles' range, the middle four raised give a round hill (a 4 x 4 grid has no
  middle dot; interpolating the dots gave a flat-topped mesa), and it meets the floor
  past its edge level, with no crease. The dots are handles, not points on the
  floor: the hill preset's 2.6 m handles make a 1.46 m hill. All 0 is cleaned to `[]`;
  `[]` and a level grid draw pixel for pixel as the flat floor did (a test holds it).
  What it touches: the floor is `Ground.faces` (`GRID_CELLS` cells each way over the
  area, quads where planar, else two triangles; the level floor round it cut at
  `GRID_OUTSIDE` only for the painter's sort), far to near by distance, each lit by
  the smooth surface's slope at its middle (the face's own normal showed the
  triangles), a picture laid on from straight above; walls stand on it, cut at the
  same places; with walls and a floor that is not level the floor inside them is
  drawn again after the walls, since a rise can stand in front of a wall's foot;
  shadows lie on it, "touching" measured above the tangent plane under the object
  (asking the patch at every mannequin vertex made a render 205 ms, the plane 59);
  the grid lies on it and drops the far side of a rise; the depth map, `floor_point`
  (Look at's Point) and the pose map follow. **Position y stays absolute**, so every
  object function that takes only `obj` (`painted_pieces`, `rigs`, `bounds`, ...) is
  unchanged and the maps cannot disagree: what moves an object re-stands it instead
  (`stand`, keeping `above_floor`) - the viewport drag, the X and Z sliders, Add,
  Enrich's `_place` - and `shape_floor` re-stands everything when the handles or the
  room's width / depth change, one undo step named "Shape the floor". The Place
  slider is "Above the floor". Not handled: an object stands by its middle, so on a
  slope one corner floats and one sinks; a crowd stands at its middle's height, so
  its people float or sink; and the room is still the backdrop, so a rise between the
  camera and an object does not hide the object (the depth map's z-buffer does).
- **What touches the floor leaves a shadow on it**, or the picture made from the frame
  draws the person hovering: the blockout was geometrically right (soles exactly at
  y 0) and still read as pasted on. `shadow_polys` adds, per object, soft nested rings
  under each part within `CONTACT_REACH` of its lowest point (each planted foot, a
  knee, a box's base - a lifted foot casts none) and a faint one under its outline up
  to `AMBIENT_REACH`. They are `Poly`s with `dim`: `rasterise` multiplies what is under
  them rather than painting, so a pictured floor stays pictured. They belong to the
  room (owner None, drawn after it, before every object), which is why an object
  standing at y > 0 casts none - its platform would be drawn over it. Tk cannot
  multiply, so each carries a flat stand-in `rgb` (the floor's colour darkened by its
  ring and those outside it) that the window draws on a plain floor; on a pictured
  floor the bake has them multiplied in, and the bake's key has two halves so that
  moving only a shadow keeps the last bake up until the new one, not a plain floor.
- **Shapes and props are meshes in a unit box.** `MESHES` holds each as parts in
  x, z -0.5..0.5 and y 0..1, each with its own colour or None for the object's (a
  tree's trunk is brown whatever colour its leaves are), scaled by the object's scale
  in metres, so a table is 1.4 m because its scale says so. Each part is one convex
  solid, since `outward` orients faces from the middle. The library lists them by
  `group`: People as buttons, Shapes and Props as a menu each - eighteen buttons left
  the scene list no room.
- **A background crowd is one object of many people.** `crowd_members` deals
  `count` mannequins into a `width` x `depth` area, at least `CROWD_SPACING` apart (as
  many as fit), each with their own height, build, skin, hair, clothes, pose
  (standing, chatting, walking, cheering) and facing, all from `seed`: the same
  settings give the same crowd, and Shuffle is a new seed. Different people in the
  frame is the point - one mannequin copied reads to the model as one person cloned.
  `wear` dresses them from outfit presets, copied on like a character's look
  (`dressed` only remembers which, for the dropdown). The words say the crowd in one
  line, "a background crowd of N people", with the user's description as written;
  the members' dealt clothes are never said, only drawn. It is not one of `people()`,
  so a scene of a crowd and props leaves the form's person in place, as props alone
  do. `crowd_pieces` is cached by its settings (a drag of anything else redraws it
  unchanged), each member is a part `m<n>` so each casts their own shadow, and a
  selected crowd is boxed rather than outlined face by face.
- **Outfit presets are a library kind.** `outfits.json` beside `characters.json`
  (`clean_outfit`, five starters in `_default_outfits` worded in the mannequin's own
  colour, shoe and hat words). A preset holds `OUTFIT_KEYS` - the Clothes and
  Accessories slots - and putting one on (`wear_outfit`) replaces all of them, so a
  slot it leaves blank comes off; body, face and hair stay. The controls are on the
  inspector's Clothes and Accessories tabs only: on every tab they pushed the pose
  sliders below the scrolled panel's visible area, and a Tk Scale that is not mapped
  never runs its command.
- **Undo is whole-scene snapshots.** `History` keeps up to 200 steps of the scene as
  JSON text, and `change_label` names each from what differs from the step before
  ("Move Crate", "Pose Ada", "Move the camera"), so no edit has to say what it is -
  a new control gets undo for free as long as it goes through `changed()` (or
  `remember_soon()` for the name and description boxes, which only retitle). A step is
  recorded once edits stop for 600 ms, so a slider dragged or a sentence typed is one
  step; a viewport drag records on release, never mid-drag. Undo and redo record a
  pending edit first. Camera moves are steps: the frame is the output. A restore puts
  the snapshot into the *same* scene dict, because a pose still being found from a
  photo checks `self.scene is scene` before landing. `dirty` after undo is compared
  with the snapshot taken at Save, so undoing back to it clears the asterisk. Ctrl+Z
  is bound on the window but passed through in a Text or Entry, which undo their own
  typing. New and Open start a new history.
- **The rig is forward kinematics over named controls.** `JOINTS` is the skeleton,
  `CONTROLS` the handful of sliders grouped by part (body, head, each hand and foot),
  `POSES` presets of them. A click on the mannequin selects the part under it (each
  face carries its object and part as canvas tags), and the inspector shows that
  part's sliders. Moving any slider by hand clears the preset name, so the words stop
  claiming "kneeling" for a pose that no longer is.
- **A photo's pose is fitted, not copied.** **From a photo…** in the Pose section sends
  the photo to the first enabled backend with `StudioDWPoseKeypoints`
  (`Studio.find_poses`: LoadImage and that node, not a job, nothing in History), which
  answers DWPose's 133 COCO-WholeBody points per person as JSON text. The node is ours,
  in `comfy_nodes/studio_dwpose` (copy it into ComfyUI's `custom_nodes`, restart), because
  it needs only what ComfyUI's venv already has - onnxruntime, OpenCV, numpy - and
  `yolox_l.onnx` + `dw-ll_ucoco_384.onnx` from huggingface.co/yzd-v/DWPose in a `dwpose`
  model folder (`D:\ComfyUI-models\dwpose` on the 5090, named in its
  extra_model_paths.yaml). It runs on the CPU on purpose, about a second a photo: the
  GPU stays the picture's (since the cu130 torch switch, CUDA would load; not used). `fit_pose` (stdlib) then searches the controls and the yaw
  for the mannequin whose joints, seen front on with no perspective and scaled to fit,
  fall on the photo's points: every facing coarsely on the head, shoulders and hips
  alone (a limb still at rest pulls the torso to make up for it), then each limb from a
  spread of starts, then all of it. A flat photo cannot say whether a limb reaches
  towards the camera or away, so the cost leans on the rest pose (`PRIOR`) and against
  arms swung back and the body leaning back (`BACKWARDS`); without that, an arm straight
  up came back as the body leaning back. The most prominent person is used (biggest box
  times score); the person is turned to face the scene's camera plus the photo's yaw;
  a part the photo does not show is left at rest and said so; the preset becomes Custom.
  Hands and feet are not fitted: the rig has no finger or ankle controls.
- **A picture makes a whole scene, stood where it stands.** **From a picture…** (beside
  New) sends the photo to the same pose finder, fits every person it sees (the most
  prominent `PICTURE_PEOPLE`, 10; the rest are said, for a background crowd), and places
  each one from how big they are: the fit reports `scale` (photo pixels a metre, from the
  whole laid-over skeleton, so a bent or turned person is not misjudged the way a box's
  height would be) and where the pelvis falls, and depth is the lens's focal length
  over that scale (`picture_scene`). The camera is level at `PICTURE_LENS` 35 mm and at
  the eye height that puts each pelvis at its own pose's height, near people weighted
  by scale squared (a far one is a few pixels and the photo's real tilt moves it most:
  unweighted, a couple's camera came out 2.8 m up instead of ~1.2). It orbits the most
  prominent person. The lens and tilt are guesses and the status says so; a synthetic
  photo of a scene comes back within centimetres. The words - the setting into Details,
  the floor, each person's name, doing and look slots - come from the host's vision
  model (`Vision.ask`), and are optional: no vision model still makes the scene.
  **The vision model is not given our numbered boxes.** Qwen2.5-VL 7B numbered them in
  its own order and put the band's instruments on the audience. It is asked for its
  own box per person (it answers in the photo's pixels, placing people across the frame
  well and up and down loosely) and `match_people` pairs each with the found person
  whose middle is nearest, across weighted over up-and-down, within `PICTURE_MATCH`;
  anyone unmatched is posed without words. Its replies also copy the example ("in
  their 30s...") and say "man" for "a man"; `_said_word` cleans both.
  **The room comes with it.** The model also says `indoors` and, indoors, what the
  walls are. Indoors, `picture_room` puts up four walls round the camera and every
  person (`PICTURE_ROOM_MARGIN` beyond the farthest, within `ROOM_LIMITS`: the room is
  centred on the origin, and a camera outside it would look through the near wall);
  outdoors there are no walls and the setting's words carry the background. Then
  `_pictured` presses Make on each surface that has words, so the floor and wall
  pictures are queued in the Image Studio as soon as the scene lands.
- **A move is on a level plane through the object's middle**, not the floor. A ray
  through a person's chest meets the floor far behind them nearly edge on, and a 60 px
  drag moved one 14 m; when even the middle's plane is edge on, the drag falls back to
  screen-space metres at the object's depth.
- **It cannot outlive its form, and it goes on the UI thread.** `Session.close` runs on
  a worker thread, and a Tk call there raises "main thread is not in main loop". So
  `ImageStudio.release()` closes the builder from `Chat._close_tab` (beside
  `browser.release()`) and from `_quit`, and `ImageStudio.close()` touches no widget.
  An unsaved scene asks to be saved on the way out.
- **A scene file is the user's work.** `*.scene.json` under `image-studio/scenes/` by
  default, written through a temp file and `os.replace`. `clean_scene` opens damaged
  or older files with what it can read and names what it could not (an unknown
  asset); objects are named by asset id, not geometry, so a better mesh from Blender
  later opens old scenes unchanged. Closing with unsaved changes asks first.
- **The room is the backdrop, and its pictures are made by the Image Studio.**
  **Floor and walls**, the list's second row, holds the floor and four optional
  walls around the origin (`scene["room"]`, `new_room`). Each surface takes a
  few words; Make sends them through `ImageStudio.generate(base=...)` as a text
  to image job built by `texture_settings` - `base` replaces the form, so the
  form's person, identities, LoRAs and references stay out of a floor - and
  `scene_texture` in the job's settings brings the finished job back to
  `SceneBuilder.texture_done` from `ImageStudio._job_changed`. Picture… puts a
  PNG from disk there instead. Either way `import_texture` keeps a 256 px copy
  under `image-studio/scenes/textures/`, named by content, so clearing History
  does not take the floor with it. It reads the full-size PNG in pure Python
  (seconds at 1024 px), so it runs off the UI thread. The words also go into
  the prompt as written ("The floor: ..."). The room is drawn before every
  object, never sorted among them, so an object outside the walls is not hidden
  by one; walls face inward, so a camera outside sees in as into a doll's
  house. `TexMap` lays a picture on a plane perspective-correctly a scanline at
  a time, with mip levels chosen by how many picture pixels a frame pixel
  spans (the far floor otherwise shimmers into moire the picture would copy).
  The viewport cannot texture a canvas polygon, so it draws the room in each
  picture's mean colour at once and puts a half-size bake (`_bake`, 150 ms
  after the last change) over the frame: a drag is never held up by one.
- **✨ Enrich offers one lived-in detail at a time.** The button asks the model the
  tabs already use (`Chat.llm`, else what LM Studio has loaded: never a new load on the
  shared card) for one detail, through `sc.suggest` off the UI thread. `ENRICH_SYSTEM`
  holds the rules: photographic imperfection and storytelling over themed clutter,
  around the people, never between them or on them, 25-60 concrete words. A different
  `ENRICH_ANGLES` kind is asked for each time. Add / Skip / Don't suggest again are
  kept in `scene["enrich"]` (`added`, `seen`, `never`, 40 each) and saved with the
  scene, so a reopened scene gets something new and a repeat is re-asked once.
  **Add puts a detail with a body into the scene** (`place_suggestion`): the model
  names a `shape` (any prop asset id, or a whole person), a true size, a colour and a `WHERE`
  word, never coordinates. The position is worked out here from the subjects' middle
  and spread along the camera's floor-level forward and right. A spot is refused if it
  overlaps something on the floor, if its frame rectangle hides behind a subject or
  covers one (`_hides`), or if it is outside the frame. The search pulls in, steps
  back, then falls back to behind-on-that-side and far background. A passer-by it
  placed is not a subject, or the middle drifts back with each one. The object's
  description carries the words; select it and drag to adjust. Light, haze, stains
  and a hand at the frame edge are `none`, and those go into the words as written
  (`added`), before the camera line.
- **Shapes are stand-ins; props are parts under one transform.** `MESHES` holds each
  shape and prop as [(faces, colour or None)] in a unit box (x and z -0.5..0.5, y
  0..1) standing on the floor: the primitives (box, cylinder, sphere, cone, frustum,
  capsule, wedge, pyramid, and `plane`, a thin box labelled Panel) and props made of
  parts (table, chair, bench, shelves, barrel, tree, bush, lamp, parasol, car, fence),
  a part keeping its own colour where it has one. So a prop is one object with one
  transform, scale is its overall size in metres (Enrich's `size` means the same), and
  a scene file names it by id, never by geometry: **never rename or drop an asset id**,
  or every scene that used it opens without it ("an object of unknown kind"). This
  library is the scene-from-picture branch's, merged 2026-09-26, with main's frustum,
  capsule, panel, shelves and stand-ins added. `STAND_INS` (Cabinet, Door, Post,
  Tree…) only start a shape named, sized and coloured through
  `new_object(..., stand_in)`; nothing about the preset is saved. What a prop is comes
  only from its name and description (`scene_text`), and the Library and inspector say
  so.
- **A face picture is drawn in twice: into the picture, then refined last.** A person's
  face picture is theirs (**Face** at the foot of their panel, copied under
  `references/scene-faces`) or their character's identity's first reference
  (`face_picture`). `face_targets` gives each seen person's face position (head joint +
  `FACE_UP`, projected), their head `region` (`FACE_REGION`, in head heights), and their
  own words (look, description, the scene's details). **In the picture itself**
  (`Studio._faces_into_picture`, `add_pulid`): one `ApplyPulidFlux` per face at
  `PULID_BASE_WEIGHT`, chained on the samplers' model, each confined by a `region_png`
  attention mask - so FLUX draws their head, hair and skin with the body. **Last**, the
  face pass (always on for a scene with people) matches SAM3's boxes to the positions
  (`match_faces`), redraws each face from its person's words plus the style (`_restated`),
  and a face with a picture with PuLID again at the scene's **Face likeness** (the
  redraw's denoise, default 0.6: on a full-length pair 0.45 left both faces thinner and
  younger than their photos, 0.6 was fuller and still seamless, 0.75 put a faint box
  round Partner's head - measured 2026-09-26).
  Why both, measured 2026-09-25: PuLID only in the last pass needed 0.85-0.92 to
  change a stranger's face into theirs, and then read as a sticker - a smooth pale face
  on a tan neck, a halo of repainted background where the stranger's bigger hair was,
  the SX-70 grain gone. With the likeness in the base picture the last pass only
  refines, and 0.45 is seamless; 0.9 put the halo back. A light whole-picture redraw
  after the faces was tried to unify grain: at 0.2 it repainted the whole wall and
  washed the likeness out - do not add one. The face pass's blend: only the oval is
  noised (`SetLatentNoiseMask`, hard-edged at `FACE_REDRAWN`; a soft noise mask left a
  pale ring), and the head is blended back, not the oval - SAM3 "head" on the redraw OR
  the original crop, OR the found face box (`FACE_BOX_GROW`, never below the chin;
  SAM3's "head" can come back as hair alone or holed over the face), inside the oval,
  softened. A face picture must show one face: PuLID takes the biggest, and the
  profile photos are of two people. Glasses and skin come from the look's words, not
  the picture. FLUX.1 only (`_pulid` says why not). ~60 s for two faces on the 5090.
- **Then their real face, but only where a photo's angle fits.** PuLID's face is *like*
  the person's, never theirs; their own pixels are them, but a front-on photo pasted over
  a turned head came out doubled (2026-09-25). So with **Real faces** ticked (the scene's
  `real_faces`, on by default) a third run, `Studio._real_faces` -> `paste_graph`, hands
  ComfyUI's `StudioFacePaste` (`comfy_nodes/studio_facepaste`) each matched face's finder
  box and every photo of that person (`face_photos`: their Face picture, then their
  identity's references). The node reads yaw and pitch of the drawn face and of each photo
  (InsightFace antelopev2, the one PuLID installs, on the CPU, ~1 s a face) and pastes only
  when the nearest photo is within a tolerance that **shrinks as the face grows**
  (`TOLERANCE`: ~22 degrees at 48 px, 8 at 160 px, 6 beyond) - a miss that vanishes at
  60 px shows on a close-up. Roll does not count; the alignment turns the photo. It aligns
  on the inner face (so the face keeps its own width), keeps the outline's hull plus a
  forehead - not hair, not ears - moves the photo's LAB mean and spread to the drawn face's,
  blurs it to the picture's sharpness, adds the grain it lacks, and feathers it in. **No
  redraw after**: a diffusion pass is what loses a likeness. Two traps in InsightFace's
  2d106 points: the outline's 33 are not numbered round the jaw (a polygon of them
  zig-zags, hence the hull), and point 16 is not the chin (the axis comes from the five
  key points). The grain is a median deviation: a spread counted glasses' edges as grain.
  The report (`ui.text`) says per face which photo, both angles, the tolerance, and why
  not; it lands in the notes and `face_detail.real`. When anything was pasted the record's
  `images` are the pasted picture first and the PuLID one beside it; when nothing was, the
  PuLID one alone. A backend without the node says so in the notes and keeps PuLID's.


- **A prop can be a picture of what it is.** Make picture (under a prop's *Looks
  like*) sends its name and description, as written, through the same
  `ImageStudio.generate(base=...)` path as the room, built by `picture_settings`:
  text to image on plain white. `scene_picture` (the object's id) brings the job
  back to `texture_done`. `import_cutout` shrinks it, `key_background` floods the
  border's colour in from the edges to make it transparent and crops to what is
  left, and the prop's `picture` then stands in for its mesh: one upright `card`,
  as tall as the prop, as wide as the picture, always turned square to the camera,
  sorted by depth among the other faces. `CutMap` skips transparent pixels, so the
  frame keeps what is behind them. A picture with no plain border is kept whole
  rather than cut to pieces. On the canvas a card is a PhotoImage scaled to its box
  (a polygon cannot wear one), cached by size. There is no 3D generation here: no
  backend has a mesh node (Hunyuan3D, TRELLIS), and a card is what a blockout for
  image to image needs.
  The depth map uses the same card and its transparent holes, including cropped
  depth maps; it must not retain the original stand-in mesh. An imported PNG with
  transparency keeps its existing cutout instead of being colour-keyed again.
- **Regional character prompting keeps each named character's words in their own
  part of the picture** (2026-09-28: the instance buffer `id_render` renders for
  masking landed the same day with nothing downstream reading it yet - this is
  that downstream). `character_masks` (`scene.py`) turns each named character's
  own instance id, from `id_render`'s sidecar, into a soft-edged binary mask
  (`_feather`, a stdlib box blur - no PIL/numpy); `scene_text_regional` pulls
  that character's line out of `scene_text`'s shared paragraph into its own
  entry, so the base text does not repeat what a masked region already says.
  Off by default (`scene["regional_prompting"]`, the Scene Builder's own
  checkbox) and a no-op under two named characters - one character is not a
  bleed problem. `imagegen.fill()`'s `regional_conditioning` block (declared
  by a workflow template, so far only `zimage_hq.json`) injects one
  `LoadImage` -> `ImageToMask` -> `CLIPTextEncode` -> `ConditioningSetMask`
  chain per character and combines them onto the base prompt's own
  conditioning with stock `ConditioningCombine` nodes - deliberately not
  ComfyUI Impact Pack, so nothing new has to be installed. A model whose
  workflow does not declare it just gets the words normally; regional
  prompting never blocks a job. `tools/ab_regional_prompt.py` runs the same
  scene, seed, model, LoRAs, pose and depth twice (regional off, then on)
  for a by-hand comparison - not yet run live.
- **Stdlib, like everything else.** The meshes are built in code, the renderer is a
  painter's algorithm with back-face culling and near-plane clipping (a prop's faces are
  cut into ~0.3 m `tiles`, or a wall running away from the camera sorts by its middle
  and is drawn over a person at its far end; a test holds that case), and the PNG is a
  scanline fill written by `core.icons.png`. Blender can make better assets offline;
  nothing here needs it to stage a picture.

### The tab that holds a window: `PanelSpec`

Milanote is a web app with no public API and no MCP server, so its tab has no model,
no bridge, no transcript and no composer. `PanelSpec` (`panel = True`, `drivable`,
`bridged` and `research` all False) is in `TABS` but, like chat, not in `APPS`.
`apps/milanote/milanote.py` does the work:

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
update. Python's stdlib has no COM client, so `apps.adobe.com.ComHost` keeps one
`powershell.exe -Sta` worker per app holding the COM object, and sends it one request
per line: the script file to run and the file to write the answer to. Things to know
before changing either bridge:

- **A tool is a script body.** `HOST.run(body)` wraps it in `apps.adobe.com.PRELUDE` (a JSON
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

- **The same wrapper as the COM bridges.** `CepHost.run()` calls `apps.adobe.com.script()`,
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
  real app — is done by hand with `python core/mcp.py check --app premiere --call`.

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
prefix a real message uses — once per tab, not once per question. It is a cost to the
*window*, though, so the shared blocks are kept terse, with a REPLIES rule for the
model's own prose.

**Big tool sets go by reference** (`core.tasks.offered_tools`). Past `LAZY_CHARS` of
schema JSON, inference gets `studio_tool_call` (whose description is a one-line index
of every bridge tool) and `studio_tool_schema` (full schemas on demand, into the
history) instead of the schemas: After Effects' ~97k chars become ~7k, Resolve's ~32k
become ~3k. `_call` unwraps a by-reference call to the tool it names, so validation,
journal, repeat checks and the transcript are unchanged; a direct call by the real
name still works, and a refusal carries the tool's schema.

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

**The modules pulled out of `core.chat`, and the rules that keep them out.**
`core.doctor` (where things are kept, the error log's one writer, the diagnostics
report), `core.files` (attachments: headers, folder listings, the copy into OpenCode's folder)
and `core.ui` (palette roles, `blend`/`rounded`/`clip`/`pretty_host`, `Pill`).
`core.chat` re-exports every name it used to define, so the rest of the app reaches
for them where it always did — but edit them in their own module.
`tests/test_doctor.py::ModuleBoundaryTest` enforces the two rules worth having: the
headless pair must import with tkinter entirely unavailable (tested by blocking it on
`sys.meta_path`, not by reading the source — a *guarded* probe inside a function is
fine and wanted, since `python_rows` reports a missing tkinter on purpose), and none
of the three may import `core.chat` back. `core.ui` is exempt from the first:
`Pill` is a Canvas.

**What is left of `core.chat` is one class, and that is the real shape of it.**
`Chat` is ~220 methods over ~3,700 lines. A mixin carve-up — `class Chat(SidebarMixin,
TranscriptMixin, …)` — would scatter the text across files while every piece still
reached into `self` state defined somewhere else, and would cost the one thing the
single file currently buys: everything about the window is findable in one place.
Do not do that. Further splitting should be real collaborator extraction, one at a
time, each owning its own state and reached through a narrow interface — the
animation engine (`_animate`/`_arm_anim`/`_anim_tick` over `anim`/`anim_timer`/
`anim_frame`) is the cleanest candidate and would make an `Animator` that takes a
widget to schedule on and a `report` callback for failures.

**`core/doctor.py` must not import tkinter.** It holds the layout of
`%APPDATA%\StudioAssistant` (`settings_path`, `data_dir`, `error_log_path`,
`tasks_dir`), the one error-log writer, and the diagnostics report — and the whole
point of `python core/chat.py --doctor` is that it answers "I clicked the shortcut
and nothing happened", which includes a Python whose tkinter is broken or missing.
Importing the GUI to ask what is wrong would fail for the reason being asked about.
`core.chat` re-exports `settings_path`, `error_log_path` and `log_error`, so the
rest of the app still reaches for them where it always did; edit them in
`core.doctor`. `--doctor` is handled before the DPI call and before
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

**Auto-update only fast-forwards, and follows `main`.** `studio_update.py` (a
scheduled task under `pyw`, set up with `--install`) polls GitHub and runs
`merge --ff-only` against remote `main` (`BRANCH`). It used to follow whatever
branch was checked out, and the workstation sat on a feature branch that had been
merged: a merged branch never moves again, so updates silently stopped. A folder on
another branch, including a merged branch or detached HEAD, now pauses updates and
explains that the user can switch to `main` when ready. The updater never switches
branches or changes tracking configuration. It uses main's configured remote, or
origin/the sole remote if main has none. It must never merge, stash, reset or check
out over local work: the workstation sometimes carries its own commits, and an
updater that "resolves" them destroys them silently. A refusal goes to
`studio_update.log`, and so does a successful update. A pass with nothing
to do writes nothing. After pulling, the pass also **pushes** (`push()`): when local
`main` has commits the remote lacks and the remote has none local lacks, `git push
<remote> HEAD:refs/heads/main`, never forced, only from `main`. If both sides moved
it pushes nothing (pull already logged why); a refused push is logged. The app's
Update button runs `pull()` only. It sets `GIT_TERMINAL_PROMPT=0` because a credential prompt with no
console hangs forever and nobody sees it.

The window has the same updater on a button. `Chat._update_tick` calls
`studio_update.check()` (fetch and compare, never a change) 8 s after start and every
15 minutes; on `main`, when remote `main` is ahead, an **Update (N)** button appears in
the header. It lists the new commits, runs `pull()` (the same fast-forward, refusals
and log as the scheduled task) and offers a restart: `main()` releases the
single-instance lock before `relaunch()` starts the new copy, or the new copy would
find the old one's lock and say it is already running. *Help → Check for updates...*
does the same on demand and also says "up to date" or why it could not tell.

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

**Corners, text size and icons are Preferences, applied live** (2026-09-28, the user:
"ui settings to adjust rounding as well as uploading icons for everything ... and
font scaling"). *Corners* is `ui.ROUNDING`, a factor every `ui.rounded` corner (and
`_round_off`'s masked picture corners) is multiplied by. Callers still pass their
designed radius, so nothing else knows about it. Past the caller's own radius it
stops at half the shape, because a smoothed polygon with a bigger radius folds over
itself. At 0 the polygon is not smoothed, since smoothing blurs the corner into a
bevel. `Pill(round=True)` discs are ovals and stay round. `_rounding` is the theme's
repaint plus `_redraw_marks`, because `_theme` never redraws badges. *Text size*
(`_scale_fonts`) configures every `tkfont.Font` the window made, plus Tk's named
fonts that other windows fall back on, to a multiple of the size recorded the first
time. Configuring a shared `Font` resizes every widget using it in place.
`_text_size` then re-runs `_metrics` for the rail's width (`side_frame`), repaints
(pills and chips measure their text) and refits the tab strip. A window that makes
its own fonts (the Image Studio's emoji) does not follow. Both are sliders
(`_slider`, shown as a percentage) over `ui.ROUNDING_RANGE` / `ui.TEXT_RANGE`. A
saved value outside its range falls back to 1.0 (`Prefs`, `ui.in_range`). The fixed
choices came first; the user asked for sliders. A drag is applied once it pauses for
`SLIDER_SETTLE_MS`, because a text-size change lays out the whole window, and doing
that at every step of a drag stutters. *Icons*: any
mark (every tab in `TABS`, bridges added by hand, every rail row) can be replaced by
an upload. `icons.upload_png` reads PNG and .ico itself and anything else through
System.Drawing (`_by_windows`, which also shrinks a big PNG before `resample`'s pure
Python sees it). It centres the picture on a transparent square, never stretching
it, and keeps a 256 px PNG under `icons/` beside the settings. The file is named by
its bytes, and the prefs hold only that bare name (a path is dropped). `_read_icons`
tries the upload first, then the .exe, then the drawn mark, and now reads every tab
at row size too, since Preferences shows them all at it. The conversion script sets
`$ErrorActionPreference = 'Stop'`; without it, a file that is not a picture ran on
through a null image and saved a blank 1x1 PNG, which read as a successful upload.
The list is its own scrolling window (`_icons_window`), not a section of
Preferences: its rows made Preferences taller than a laptop screen, and Preferences
does not resize.

**Buttons take icons too, by label** (the user: "let me upload icons for all buttons as
well", then "hide text, show text, only show the icon"). A text button's icon is
keyed `button:<label>`, so the Send of every tab is one upload. A glyph button's is
keyed `glyph:<name>` (`GLYPH_NAMES`: add, close, pin, unpin, folder, more). Both kinds
live in the same `icons` pref and folder as the app marks. `Pill.paint` asks its
own window (`self._root()`) through `pill_icon`, `pill_show` and `pill_rename`. It
gets the picture (the text's line height, drawn before the label), a `BUTTON_SHOWS`
mode (text, both or icon; both is the default and is not saved) and the renamed
words. Icon alone drops the words only when there is a picture, so no button can be
blank. **Nothing is held on the `Pill` class.** Class-level hooks came first. They
pointed at whichever window was made last, so a test that made and closed a second
window left every button asking a dead one. They also kept a closed window's Tk
images alive until Python freed them on a worker thread, and "Tcl_AsyncDelete:
async handler deleted by the wrong thread" killed a full test run. Glyph labels are registered in `glyphs`
(`_glyph_icon`) and show the picture in place of the character. The pictures are made
on the UI thread from the kept PNG (`_icon_photo`, cached by key, size and file name).
They are not read by the icon worker, because a button's size follows the text size.
`_repaint_buttons` redraws both kinds after an upload, a Reset, a mode change or a
text-size change. The Icons window lists the labels of the pills alive at the moment
it opens (`_button_labels`), plus any label that already has an icon.

**Anything in the Icons window can be renamed, and the connection rows take icons**
(the user: "the connections does not let me change icons. i would like to be able to
change the text of items within the icon changer"). Every row but a glyph's has its
name in an entry. Enter or leaving the field saves it (`_rename`, the `names` pref,
key -> words); blank or the program's own name takes the rename away. Reset puts
back all of icon, name and show. The names are for display only: `Pill.text`, a
row's `a["name"]` (which pins and hides are keyed by) and `app.name` in messages stay
the program's. What shows the new name: tab chips, heroes, rail rows, the new-tab
and bridges menus, every button (`pill_rename`) and the Inference and Bridges rows
(`CONN_NAMES`, keys `conn:host` / `conn:bridges`). A plain label is "dressed"
(`_dress`, registered in `dressed`) with its key's name. With `icon=True`, as on the
connection rows, it also shows the uploaded picture before the words, and Text / Both
/ Icon applies there as it does on a button. The status dot or arc stays, because it
is state and not decoration. `_repaint_buttons` redresses them all. A rename also
rebuilds the rail and refits the tabs (`_renamed_everywhere`).

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
The user's chat window mid-run. Test by constructing `Chat()` in-process and calling
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
records, OpenCode's config), `studio_assistant_error.log`, `studio-assistant.ico`,
and the CEP panel's `ExtensionBundleId`. The panel's *display* name did change, so
`python apps/adobe/premiere.py --install-panel` has to be re-run for the entry under
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
started with `core.procs.spawn()`, never a bare `Popen`. Read the module docstring
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
  by COM's own service. Neither is ever in a job of ours. OpenCode's server *is* ours
  (`ServerSpec`): it is spawned into a job and ends with the window.
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
- **Every tab has a log of its own** (2026-09-28, the user: "add logs to every tab").
  The header's **Log** button, File ▸ Log for this tab… and Ctrl+L open a window
  of the current tab's lines, live (`_log_window`). The lines are the activity
  log's (`studio_activity.log`), filed by tab: `core.tablog` gives every record a
  `tab` (`Stamp`), the file prints it as `[app id]` (`-` for the window's own), and
  `tablog.BOOK` keeps each tab's last `KEEP` lines in memory, so a Log window
  opened late still shows what came before. A record gets its tab two ways. First,
  `_log_event` writes each tab's events from `_handle`, the one place every tab's
  events pass: sys/error lines, calls and results, status and bridge
  *changes*, replies, questions, a panel tab's notes, an Image Studio job's status
  changes. Nothing frequent is written (no tokens, step counts or terminal
  screens). Second, `_guard` runs a tab's worker in `tablog.working_for(tab)`,
  so `studio.agent`'s tool-call and model-request lines land in the tab that made
  them without the engine knowing tabs exist. That is a thread-local, and **a
  thread started inside a tab's work does not inherit it**. `MCPClient` reads
  `tablog.current()` in `__init__` and re-enters it on its stderr thread, which
  is how a bridge's stderr (`studio.bridge`) reaches its tab. Before this, that
  stderr went to a console `pythonw.exe` does not have. Log lines reach an open
  window through the queue (`("log", tab, entry)`), never from the logging thread
  to a widget. `BOOK` takes one listener per window, since the tests build
  several and a second must not silence the first. Tests: `tests/test_tablog.py`,
  the `tab logs` tests at the end of `TestGui`.

## Task execution and recovery

- GUI and CLI use `core.tasks.Executor`; do not add another tool loop. That
  includes the chat tab: same executor, same journal, an empty tool list.
- Keep original MCP schemas in `Session.schemas` for validation. Sanitized schemas
  are the inference representation only; preserve compound-tool descriptions.
- `studio_task_update` is an internal tool exposed alongside bridge tools. Warm-up
  must use the same system prompt and tool list, including that internal tool.
- **Continuation is automatic and referable.** With `studio_task_recall` exposed,
  `context_messages` archives whole older exchanges as `record.notes` before they
  leave the request. Notes contain bounded source excerpts and the complete source
  messages, under stable `note:N` references; full live history is also retained.
  The executor checkpoints new notes before asking the model again. No model call
  is required to summarize, and excerpts are historical data, not instructions or
  proof of an edit. A compacted request carries the original request note and recent
  notes; `ref=index` searches their summaries, `ref=note:N` pages the source,
  `ref=journal:N` pages full saved tool evidence, and `ref=state` pages all briefs,
  plans, objects, checks and issues. Oversized state excerpts show the latest request
  first and explicitly require retrieval of omitted constraints. Recall never
  clears a restore guard or read-back obligation. Old task files restore with an
  empty note list and compact normally on their next run.
- **The input budget follows the loaded model.** Each executor run probes the
  model's loaded context after settling GPU work, reserves up to 4096 output tokens,
  and uses a conservative 2.5 input characters per remaining token, capped by the
  caller's budget. This is an estimate, not exact tokenization. A host that does not
  report its window uses a 32000-character fallback. Output reservation is sent as
  `max_tokens` in both streaming and non-streaming requests. A context/length refusal
  permits one retry with a smaller context; partial calls never execute. If the
  fixed prompt and tool contract cannot fit, fail explicitly rather than clip them.
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
  Since 2026-09-27 `studio_remember` and `studio_tool_create` count too
  (`BOOKKEEPING`). Seen live, a finished model called one after another, rewriting
  its answer each time, until the user stopped it. The texts were paraphrases (0.06-0.29
  similar), so comparing answers would not have caught it. Open roadmap steps still
  continue.
- **The roadmap** (added 2026-09-27; the user: "it needs a roadmap of tasks to finish
  the prompted task"). `TaskRecord.plan` is a list of steps and `TaskRecord.done`
  holds their numbers (`studio_task_update` takes `done`; a new plan clears it).
  `roadmap()` renders a checklist ending in "Next: step N." It is in the tail of
  every request (`context_messages`) and in the reply to each update. The GUI shows
  a `roadmap` event as a ✓ / → / ○ block. **The executor makes the roadmap itself
  from a brief written as numbered steps** (`numbered_steps`: 1, 2, 3… in order,
  two or more). Seen live: qwen3-coder never recorded a plan even when the prompt
  asked, though it worked such a list in order. With the roadmap seeded, it ticked
  the steps and finished at 3 of 3. When the model stops with a step open (not a
  question, no `studio_ask`), `ROADMAP_HINT` names the step, at most `ROADMAP_NUDGES`
  (2) times a run. A third stop ends the run on the model's own words. A saved task
  from before `done` existed loads with `done = []`.
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
  kept. **A streamed request is refused differently:** LM Studio answers `200 OK`
  with an `event: error` line whose data is `{"error": …}` and has no `choices`, so
  `_open` never sees an HTTP error. `LLM.stream` reads that line. When it comes before
  any content and a draft is set, the draft is dropped with the same note and the
  request is retried once. Otherwise it raises the host's message. Before this was
  fixed (2026-09-27, `qwen3-1.7b` refused as needing "load time"), every reply in the
  tab read "Incomplete inference response (connection ended)".
  `_draft_check` says the note once, in the tab the request was made from, after
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
  tab, never a stop. Pictures go through `core.icons.flatten_png` first: a vision
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
  served tab is refused anything outside its folder: `attachment_note` copies a file
  (or a folder, whole) from elsewhere into `<workspace>/.studio-attachments/` and names
  every attachment by its path in that folder.

## What the model learns, asks and looks up

Four mechanisms, all in service of one fact: the executing model is small and
starts every session knowing the app in general and nothing about this studio,
this bridge's failures, or what the user said last time.

### The research sidecar: files and the web on every tab

`apps.research.mcp.SERVER` — the Chat tab's bridge — is offered to every app tab
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

`core.lessons.Notebook` is `lessons/<app>.json` beside the settings, loaded by
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
- **Lessons come in layers** (added 2026-09-27, the user asked for OpenCode that
  "becomes smarter every answer"). `core.lessons.for_app(app)` is a `Stack`:
  `everywhere` (`lessons/_everywhere.json`, every tab), `app` (`lessons/<app>.json`),
  and, for an app with a `workspace` (OpenCode), `folder`
  (`lessons/folders/<name>-<hash>.json`). New lessons go to the most specific layer.
  The exceptions are "remember everywhere: …" and `studio_remember` with
  `scope: "everywhere"`, which go to the global layer, and refused calls, which go
  to the app layer. A lesson already kept in any layer counts as a repeat. `Stack`
  answers the `Notebook` interface, so the executor and the learner never see layers.
- **One notebook per file per process** (`core.lessons.shared`). Tabs share the
  `Notebook` objects; the record of what a prompt already carried lives on each tab's
  `Stack` (`_carried`), not on the notebook. When each tab loaded its own copy (seen
  live), a lesson kept "everywhere" in the OpenCode tab never reached the Chat tab,
  not even on New chat, and whichever tab saved last would have overwritten the other.
  Additions and forgets hold `_LOCK`, because the tabs' workers share the notebooks.
- **A message that only states a lesson is not a task.** `Chat._remember_turn` runs
  before Direct and before the executor. It keeps "remember …" / "from now on …" in the
  user's words and answers "Kept for <layer>: …" with no model and no OpenCode call.
  Seen live: when the model got "remember run tests with pytest -q", it made a tool,
  called studio_remember with a paraphrase and cycled on plans until the user stopped it.
  The stop then skipped `_learn`, so the user's own lesson was never kept.
- **OpenCode reads the lessons too.** `Chat._publish_lessons` writes the stack to
  `OPENCODE_STATE/lessons.md` (`ServerSpec.lessons_path`). It is written at tab boot,
  at the end of every turn (`_turn`'s `finally`, which also covers studio_remember
  and stopped runs), on a remembered message, and on each forget. `write_config`
  adds the file to `instructions`. **OpenCode 1.18.32 re-reads `instructions` on
  every prompt, not only at startup.** This was tested live with canary words that
  reached OpenCode only through that file: one was learned after the server started
  and was answered from a new session with no restart; the other was added between
  two prompts of the *same* session and was answered on the second prompt. A
  forgotten lesson is gone from the next prompt. No restart is needed for a lesson
  to reach OpenCode. There is no reflection in Direct mode, because no chat model
  runs there.
- **Seen live, the refused-call path needs a model that sends bad arguments.**
  qwen3-coder-30b fixed a bad `limit` and an unknown key on its own, then claimed the
  call had been "rejected", which it hadn't been. The live check of that path used
  one scripted bad call; the validation, the learning and the routing were real.
- `studio_ask` and `studio_remember` are `INTERNAL_TOOLS` with the task-record and
  tool-maker tools, in that order, before the made tools; `toolsmith.reserved()`
  already refuses `studio_` names. The Lessons window (`File > Lessons for this
  tab...`) is rebuilt on every forget; a forget while the tab is busy is ignored,
  because the worker may be writing the notebook.

### Ideas for updates: `studio_idea`

The user asked (2026-09-29) for a list of ideas for updates to the app, then for the
local models to add to it and for Claude Code to read it. `core/ideas.py` keeps
`ideas.json` beside the settings; `Help > Ideas for updates...` (`core/chat_ideas.py`)
shows it. An idea is open, done or dropped; dropping asks "why not?" in a field under
the line, because the why is what stops the idea coming back. Only Delete removes one.

- `studio_idea` is the last of the `INTERNAL_TOOLS` (so adding it changed every tab's
  tool prefix once) and counts as `BOOKKEEPING`. It is for a limit of *the app* - a
  missing bridge tool, a workaround - never the user's project. The Executor takes
  `tab=` (the app's name) so the idea says which model added it, and refuses a third
  idea in one run (`MODEL_IDEAS_PER_RUN`).
- `Ideas.suggest` never adds the same idea twice (compared after `normal`), and a
  model suggesting one the user dropped is told it was turned down and why.
- The window and each tab's worker hold their own `Ideas`. Every edit re-reads the
  file under `ideas.LOCK` before writing (`_edit`), otherwise the window would save
  its old list over an idea a model had just added. A file that could not be
  *opened* aborts the edit; only one that is not JSON is set aside as `.broken`.
  The open window checks the file's stamp every 2 s and redraws, but not while a
  "why not?" field is open.
- The list is user data, not in the repo, so git and the updater never touch it.
  Claude Code sessions read it at `%APPDATA%\StudioAssistant\ideas.json`.
- Tests: `tests/test_ideas.py`, `test_studio_idea_adds_to_the_users_list_marked_with_the_tab`,
  `test_the_ideas_window_adds_drops_with_a_why_and_reopens`.

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
`core/toolsmith.py` holds the definition, the per-app library and the checks.

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

`core/mcp.py` is the one place the protocol lives. A bridge written here is a table
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
- **A bridge can ask the user (elicitation).** `core.mcp.elicit(message, schema,
  meta)` from inside a tool sends `elicitation/create` and blocks until the client
  answers; `serve()`'s reader routes the reply (`deliver`) while the main thread waits,
  and a cancel of the tool call ends the wait as `cancel`. It raises `Declined` unless
  the client declared `capabilities.elicitation` - `MCPClient` and `Loopback` do only
  when `on_elicit` is set before `initialize`. `MCPClient` answers on a thread of its
  own and stops the call's clock while a question is open, so a user who takes five
  minutes is not a timeout. The GUI's `_elicit` shows `_show_elicit`'s card (diff box
  from `_meta["studio/approval"]`, a button per enum value, entries for strings, Stop)
  and treats `s.cancel` or a closed tab as `cancel`; the CLI's is
  `elicit_at_terminal`. This is how a tool that must not act on the model's say-so
  gets the user's.
- **Two kinds of "no".** Unknown tool, arguments the schema refuses, a parse error, a
  JSON-RPC batch: protocol errors, with the JSON-RPC code the spec names, because a
  caller that sends them skipped the executor's own validation. A tool's own refusal
  — `ComfyError`, `OpenCodeError`, the classes a bridge lists in `errors=` — is an
  `isError` result in the bridge's words, and any other exception is an `isError`
  result naming it with the traceback on stderr. The server keeps serving through
  all of it. The bridges' Python-level `call_tool()` folds the first kind into a
  result too, for callers that are not on the wire.
- **The validator is shared.** `core.mcp.validate` is what the executor runs
  before a call and what a `Server` runs on arrival; `core.tasks.validate` is the
  same function. One validator, so the two sides cannot disagree about a schema. It
  caught a test calling `comfy_generate` with a `timeout` under the schema's minimum
  the day it went in. **Its refusals teach**: an unknown key names the keys the
  object takes and the closest one (`compID ... did you mean compId?`), a wrong
  type or range quotes the value sent and the property's own description (which is
  where the units live), and the unknown-key check runs before the required-key
  check so a misspelling is reported as one. The reader is a 3B model with one more
  try; `is not allowed` on its own sent it guessing again, and the guess landed in
  `failed_calls`. Keep every new refusal in that shape.
- **A path from the model goes through `core.mcp.local_path()` before the
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
- **Progress and cancellation are opt-in per tool.** `core.mcp.progress()` sends
  `notifications/progress` on the client's `progressToken` and does nothing without
  one; `core.mcp.cancelled()` is True once the client sent `notifications/cancelled`
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
- **Installed bridges are held to recordings.** `python core/mcp.py snapshot --app
  <id> tests/contracts/<id>.json` writes what the bridge exposes, and
  `tests/test_mcp.py` checks every recording against the registry's groups and
  prompts — offline, with the app closed. Re-record when a bridge updates; a
  recording older than the bridge is a test that passes for the wrong reason. The AE
  recording shows 14 tools no group exposes (markers, house style, jobs, the issue
  journal, `delete_comp`, `init_project`, `setup_panel`); some of that is deliberate
  and the rest is a decision nobody has made yet — the check will keep saying so.

## Running and testing

### Finishing and recovery

- Scene Builder's close result can veto tab/app closure. `ImageStudio.can_close()`
  checks before any tab state is removed or `Chat.closing` is set; `release(confirmed=True)`
  only destroys windows after those checks. Failed saves and cancelled Save As dialogs
  must leave the app usable. Settled edits write separate scene recovery copies under
  the Image Studio library; File → Recover scene opens them without replacing originals.
- Final FaceFusion targets carry the Scene Builder identity and normalized region.
  Positional form selections require the expected face count. Ambiguous detections
  must fail with a request to choose the face, never default to the biggest face.
- Face-only fixes and saved finishing retries use a local queue lane, without any
  ComfyUI health checks. A generated picture and profile snapshot are persisted before
  the final face swap; failure leaves that checkpoint available in History.
- `JobQueue.cancel()` may be called by Tk: set its event immediately, send network
  interruption on a worker, and let the UI show Cancelling until the job settles.
- Repeat seed uses current library records and model files. Do not describe it as
  exact recipe replay. `tests/test_finish_line.py` covers the recovery boundaries offline.
- **A test that starts a lane collects first.** Tk things an earlier test left in a
  cycle are freed by whichever thread the collector next runs on. On a lane's thread
  that is a Tk call off the UI thread (`tkinter.Variable.__del__`), and the lane stops
  where it stands: one job left "queued" for good, in a full run only and on the same
  test every time, because where the collector runs follows how much was allocated
  before it (2026-09-30; found with `faulthandler.dump_traceback(all_threads=True)`
  where `settle` gave up). `TempStudioMixin.setUp` calls `gc.collect()` on the test's
  own thread before its Studio starts one. The app has the same trap only if a Tk
  object is dropped in a cycle; see "Nothing is held on the `Pill` class".

- **Run the tests unseen.** The GUI tests open real Tk windows (a Chat, a Scene
  Builder, consoles); run as plain `python -m unittest` on the user's PC, every one
  appears over their work and takes the keyboard. `python tests/offscreen.py <unittest
  arguments>` runs the same tests on a Windows desktop of their own that is never
  switched to: same windows, same results, nothing shown. Use it for every run that
  reaches a GUI test, the whole suite included. It is not a test mode: the code under
  test cannot tell, and nothing is skipped.
- **A GUI test collects its own garbage, on Tk's thread.** A dropped widget holds bound
  methods of itself, so a closed window, rebuilt buttons or a second Chat is Tk
  objects in cycles, freed by the collector on whichever thread it next runs. In the
  app that is harmless: `mainloop` runs, and a free from a worker is handed to it. In
  the tests nothing runs `mainloop`, so each Tk variable freed off the main thread
  waits 1 s and gives up ("main thread is not in main loop"). A later module's job lane
  once stood 60 s freeing 55 of them, and `test_finish_line` failed "jobs did not
  finish" in full runs only, on a different test each time (2026-09-30; found by
  dumping every thread where `settle` gave up, and by collecting with
  `gc.DEBUG_SAVEALL` after each test to see who left Tk objects behind). `TestGui`
  (test_agent) was the only class that did, and its `setUp` now registers
  `gc.collect` as its first cleanup. A new test class that builds and drops Tk windows
  does the same.

```bash
python tests/offscreen.py discover -s tests -v   # no network, no apps needed, no windows shown
python core/agent.py --list-groups         # registry sanity, no bridge started
python core/agent.py --app resolve --list-tools   # needs the Resolve venv
python core/agent.py --app comfyui --list-tools   # no ComfyUI needed for the list
python apps/comfyui/mcp.py --list-tools      # the bridge's own contract
python apps/comfyui/mcp.py --check           # ...held to the harness's checks
python apps/opencode/mcp.py --list-tools   # likewise; no OpenCode needed for the list
python apps/adobe/photoshop.py --check       # the COM bridges; no app is touched by --check
python apps/adobe/illustrator.py --list-tools
python apps/adobe/premiere.py --check        # the CEP bridge; no app is touched by --check
python apps/adobe/premiere.py --install-panel   # copy premiere_panel/ under CEP/extensions (Premiere closed)
python apps/research/mcp.py --check        # the Chat tab's bridge; reads nothing by itself
python core/mcp.py check --app chat --in-process --call   # ...and list_folder on the home folder
python core/agent.py --app chat "find the brief in my Documents folder"   # the chat tab from the CLI
python core/agent.py --app resolve "look up the ProRes flavours and say which to deliver in"   # the sidecar in an app tab
python core/mcp.py check --app premiere --in-process
python core/mcp.py check --app photoshop --in-process   # against the registry entry
python core/agent.py --mcp "npx -y some-mcp" --name Blender --list-tools  # any bridge
python core/mcp.py check --app comfyui     # start a registry bridge, report on it
python core/mcp.py check --app after-effects --call   # ...and call its harmless reads
python core/mcp.py check --app resolve --snapshot tests/contracts/resolve.json
python core/mcp.py snapshot --app resolve tests/contracts/resolve.json  # re-record
python core/chat.py                        # the real app (console attached)
python core/chat.py --doctor               # no window, no lock: runs beside a live copy
python core/doctor.py                      # the same report, on its own
```

`tests/` never touches the network, the creative apps, OpenCode, or the model — the
research bridge's `urlopen` is swapped for a fake with a table of pages,
`test_lessons.py` drives the notebook, the reflection and the question form with the
same fake inference `test_tasks.py` uses and a temp notebook directory,
the COM bridges are tested with `HOST.run` replaced, the one PowerShell worker a test
starts is given a ProgID nothing answers to, and the Premiere bridge talks to a fake
panel on a random loopback port —
`test_comfy.py` and `test_opencode.py` swap `urllib.request.urlopen` for an in-memory
server that answers the routes the bridge uses - the OpenCode one plays a session out
step by step, stopping on permissions and questions - and works in a temp folder. `test_mcp.py` drives both bridges through `Loopback` — the real server
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
  `core/icons.py`, with the drawn two-letter badge as the fallback whenever that
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
