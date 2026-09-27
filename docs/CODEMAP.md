# Code map

Where things are, so you open one file at the right function instead of reading
around. Symbols, not line numbers: grep `def name` / `class Name` to land on one.
Rules and the why behind them are in `AGENTS.md` (long - grep it, don't read it).
Update this page when you add, split, move or rename a module.

## Layout
```
studio_chat.py          launcher: the Start Menu shortcut runs this; the app is core/chat.py
studio_update.py        the updater (a scheduled task runs this path - do not move it)
make_icon.py
core/                   the app shell and the engine every tab shares
apps/
  adobe/                Premiere, Photoshop, Illustrator bridges + COM / CEP transports
  image_studio/         the Image Studio tab
    addons/             LoRAs per model, CivitAI / Hugging Face catalogs, custom nodes
    scene/              the Scene Builder
  comfyui/              the ComfyUI tab: bridge and Nodes view
  opencode/             the OpenCode tab: bridge and coding Add-ons
  milanote/             the Milanote tab
  research/             the Chat tab's files + web tools
tests/  tools/  docs/  comfy_workflows/  comfy_nodes/  recipes/  premiere_panel/
```
- Every folder is a package; import by full name. Most modules are imported under
  their old flat name, so the code body reads the same:
  `import apps.image_studio.imagegen as studio_imagegen`.
- A module that also runs as a script (a bridge, `core/chat.py`, `core/agent.py`...)
  starts with an `if __package__ in (None, ""):` guard that puts the checkout on
  `sys.path`, so `python apps/adobe/premiere.py --check` works from anywhere.
- `HERE` / `ROOT` in a moved module is the checkout root, not the module's folder.
- Mock by the new name: `patch("apps.image_studio.facefusion.swap")`.

## core/ (every tab)
| File | Lines | What it is | Start at |
| --- | --- | --- | --- |
| `core/chat.py` | 5.3k | The Tkinter window: tabs, chat, approval cards | `Chat`, `_drain`/`_handle`/`_report`, `_elicit`/`_show_elicit`/`_diff_box`/`_settle_elicit` |
| `core/agent.py` | 3.7k | Engine: app registry, LLM client, MCP client, prompts, CLI | `APPS`, `AppSpec`, `ServerSpec`, `LLM`, `MCPClient`, `Router`, `*_PROMPT` |
| `core/mcp.py` | 1.2k | MCP harness every bridge is built on | `Server`, `tools_from_table`, `elicit`, `Declined`, `progress`, `cancelled`, `Loopback` |
| `core/tasks.py` | 0.9k | Task execution and recoverable task records | |
| `core/lessons.py` | 0.3k | What the model learns after each task, per app | |
| `core/appinfo.py` | 0.2k | Each app tab's profile for its prompt: name, installed release, Wikipedia overview (cached in `appinfo/`) | `refresh`, `render`, `WIKI` |
| `core/toolsmith.py` | 0.4k | Tools the model writes for itself from its own bridge tools | |
| `core/ui.py` | 0.3k | Palette roles and drawing primitives | |
| `core/procs.py` | 0.3k | Child processes that die with the parent | |
| `core/files.py` | 0.1k | Attachments described for a model | |
| `core/icons.py` | 0.4k | App icons pulled from each program's .exe | |
| `core/terminals_ui.py`, `core/consoles.py` | | The Terminal tab: consoles found, hidden and mirrored | |
| `core/doctor.py`, `studio_update.py` | | Where things are kept / health; sync with GitHub | |

## apps/adobe/ (one MCP stdio server per app)
`apps/adobe/premiere.py`, `apps/adobe/photoshop.py`, `apps/adobe/illustrator.py`.
Transport: `apps/adobe/com.py` (COM) and `apps/adobe/cep.py` (CEP panel, `premiere_panel/`).
After Effects and Resolve use outside npm servers.

## apps/image_studio/ (ComfyUI)
- **Tab and engine** - `apps/image_studio/ui.py` (tab, 5.9k) over
  `apps/image_studio/imagegen.py` (engine, 6.7k). `python apps/image_studio/imagegen.py --probe`
  is the first diagnostic.
- **Critic** - `apps/image_studio/critic.py`.
- **Identity** - `apps/image_studio/lora_train.py`, `apps/image_studio/facefusion.py`, `tools/`.
- **Model sources** - `apps/image_studio/model_sources.py`.
- **scene/** - Scene Builder: `scene/ui.py` over `scene/scene.py`, `scene/mannequin.py`,
  `scene/pose.py`.
- **addons/** - `addons/catalog.py` (LoRAs per model, uninstall), `addons/civitai.py`,
  `addons/hub.py` (Hugging Face, GitHub node plugins), `addons/nodes.py` (bundled
  `comfy_nodes/` installer), `addons/discovery.py`.
- Workflows: `comfy_workflows/`, `recipes/`; custom nodes `comfy_nodes/`.

## apps/comfyui/
Bridge `apps/comfyui/mcp.py`. Nodes view: `apps/comfyui/view.py` (the browser window),
`apps/comfyui/nodes_ui.py` (the Chat | Nodes switch).

## Other apps
Milanote: `apps/milanote/milanote.py`. Research (Chat tab: files + web, read-only):
`apps/research/mcp.py`, run in-process through `core.mcp.Loopback`.

## apps/opencode/, end to end
```
Chat tab (core.chat)  --model briefs-->  apps.opencode.mcp (bridge, stdio)
   ^  approval card                          |  HTTP + Basic auth (server.key)
   |  (_show_elicit)                         v
   +---- elicitation/create <---- settle() <-- opencode serve (ServerSpec.launch)
                                              |  edits the workspace (this repo)
                                              +-> LM Studio on the LLM PC
```
- **Start/stop, config** - `core/agent.py`: `ServerSpec` (`launch`, `stop`,
  `write_config`, `fit_window`, `model_for`), `opencode_config`, `OPENCODE_PERMISSIONS`,
  `OPENCODE_BRIEF` (= `docs/OPENCODE.md`), `own_repo`, `opencode_exe`,
  `OPENCODE_GROUPS`, the `id="opencode"` entry in `APPS`, `OPENCODE_PROMPT`.
  State (config, password, `last_session`, log) in `%LOCALAPPDATA%\StudioAssistant\opencode`.
- **Bridge** - `apps/opencode/mcp.py`: `t_ask` -> `start_task` (git worktree per
  task, `tasks.json`) -> `prompt` -> `finish` -> `follow` (woken by `Events`, the
  `/event` stream) -> `settle` (permissions + questions of the session `Family`;
  `granted`/`add_grant` keep "always") -> `ask_permission` / `ask_question` ->
  `core.mcp.elicit`; `report` + `context_report` summarise; `after_ask` runs
  `tests_for` and makes a `checkpoint`. `t_merge` / `t_undo` / `t_discard` ask through
  `confirm`. Every session call passes `task_dir(sid)` as `?directory=`. `ROUTES` is
  every server route used.
- **Approval UI** - `core/chat.py`: `_elicit`, `_show_elicit`, `_diff_box`, `_settle_elicit`.
  Direct mode: `_toggle_direct`, `_direct_turn`.
- **Add-ons** - `apps/opencode/codeaddons.py` (records, `config`, MCP registry / npm / skills
  search and install) and `apps/opencode/codeaddons_ui.py` (`AddonsWindow`).
- **Tests** - `tests/test_opencode.py` (`FakeOpenCode` plays a server),
  `tests/test_codeaddons.py`, `TestElicitation` in `tests/test_mcp.py`.

## Tests and ledgers
`tests/test_<module>.py` per module (named after the old flat module name);
`python -m unittest tests.test_<module>`.
`tests/VERIFIED.md` - what already passed (don't re-test); `tests/LIVE_ACCEPTANCE.md`.
