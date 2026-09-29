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
  phone/                chat and pictures as a web page for a phone (its own process)
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
| `core/chat.py` | 4.8k | The Tkinter window shell: `__init__`, layout, tabs, the transcript, host/model fit, the send/turn pipeline, the event pump. Inherits the mixins below, so `Chat`'s full method list is split across all of them | `Chat`, `_drain`/`_handle`/`_report`, `_turn`, `_boot_host`/`_fit`/`_make_room` |
| `core/chat_theme.py` | 0.1k | Applying the palette; repainting drawn (not `config()`-able) widgets on a theme switch | `ChatThemeMixin`, `_theme`, `_skin`, `_redraw_marks` |
| `core/chat_widgets.py` | 0.6k | The drawing primitives every window is built from - buttons, fields, marks, dots, arcs, menus | `ChatWidgetsMixin`, `_button`, `_entry`, `_dots`, `_arc`, `_mark` |
| `core/chat_updates.py` | 0.1k | Checking GitHub for updates and pulling them | `ChatUpdatesMixin`, `_check_updates`, `_on_update` |
| `core/chat_icons.py` | 0.4k | Preferences > Icons: reading an app's icon from its .exe, upload/reset, the icons window | `ChatIconsMixin`, `_read_icons`, `_icons_window`, `_upload_icon` |
| `core/chat_bridge_dialog.py` | 0.2k | Connect an MCP bridge by hand: the dialog and its registry writes | `ChatBridgeDialogMixin`, `_bridge_dialog`, `_save_bridge` |
| `core/chat_ask.py` | 0.1k | `studio_ask` as a form in the transcript; the card chrome `chat_elicit.py` shares | `ChatAskMixin`, `_show_ask`, `_form_card`, `_place_form` |
| `core/chat_elicit.py` | 0.2k | A bridge's MCP elicitation, answered by the user, never the model | `ChatElicitMixin`, `_elicit`, `_show_elicit`, `_diff_box` |
| `core/chat_lessons_studio.py` | 0.1k | The studio brief editor and the per-app lessons window | `ChatLessonsMixin`, `_studio_window`, `_lessons_window` |
| `core/chat_diagnostics.py` | 0.1k | The Diagnostics window: the same facts `--doctor` prints, read live | `ChatDiagnosticsMixin`, `_diagnostics_window`, `_refresh_diagnostics` |
| `core/chat_ideas.py`, `core/ideas.py` | 0.4k | Help > Ideas for updates: what to build or fix next in the app, open / done / dropped with a why, added by the user or by a tab's model through `studio_idea` (`IDEA_TOOL`, run in `Executor._call`). Kept in `%APPDATA%\StudioAssistant\ideas.json` - read it before proposing work, and do not propose a dropped idea again | `ChatIdeasMixin._ideas_window`, `Ideas`, `Ideas.suggest` |
| `core/agent.py` | 1.9k | Engine core: app registry, LLM client, GPU/model fit, OpenCode plumbing. Re-exports the split-out files below, so `core.agent.X` still finds everything | `APPS`, `AppSpec`, `ServerSpec`, `LLM`, `fit_model`, `make_room` |
| `core/agent_mcp_client.py` | 0.4k | MCP stdio transport: the JSON-RPC client every bridge is built on | `MCPClient`, `sanitize_schema`, `to_openai_tools`, `HostUnreachable` |
| `core/agent_prompts.py` | 0.7k | Every app's system prompt, and the shared rule blocks - pure string data | `*_PROMPT`, `BASE_RULES`, `CHAT_RULES` |
| `core/agent_studio_brief.py` | 0.1k | The studio brief (About this studio...) and the research sidecar (files/web, read-only) | `studio_brief_path`, `read_studio_brief`, `Router`, `research_client` |
| `core/agent_bridges.py` | 0.3k | Bridges entered by hand (`BridgeSpec`) and detecting what apps are installed | `add_bridge`, `remove_bridge`, `detect_apps`, `BridgeSpec` |
| `core/agent_cli.py` | 0.4k | The console-mode CLI: elicitation at a terminal, one turn, the interactive REPL, argparse | `main`, `converse`, `run_agent`, `ask_at_terminal` |
| `core/mcp.py` | 1.2k | MCP harness every bridge is built on | `Server`, `tools_from_table`, `elicit`, `Declined`, `progress`, `cancelled`, `Loopback` |
| `core/tasks.py` | 0.9k | Task execution and recoverable task records | |
| `core/lessons.py` | 0.5k | What the model learns after each task, per app; `trouble_in`/`self_review` for the trainer's cross-task check | |
| `core/appinfo.py` | 0.2k | Each app tab's profile for its prompt: name, installed release, Wikipedia overview (cached in `appinfo/`) | `refresh`, `render`, `WIKI` |
| `core/toolsmith.py` | 0.4k | Tools the model writes for itself from its own bridge tools | |
| `core/ui.py` | 0.3k | Palette roles and drawing primitives | |
| `core/procs.py` | 0.3k | Child processes that die with the parent | |
| `core/tablog.py` | 0.1k | Each tab's log: records stamped with their tab, last lines kept per tab | `working_for`, `Stamp`, `BOOK`; the window is `Chat._log_window`, fed by `Chat._log_event` |
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
- **Critic** - `apps/image_studio/critic.py` (`analyze_generated_image`,
  `plan_next_refinement`; the faults: `user_faults`, `score_fixes`, `carry`, `harder`),
  run by `Studio._refine` - after Generate, and after a fix with `fix["check"]`.
- **Identity** - `apps/image_studio/lora_train.py`, `apps/image_studio/facefusion.py`, `tools/`.
  Angles / Blend (new reference photos by FLUX Kontext): `apps/image_studio/blend.py`
  (`angle_graph`, `blend_graph`, `route`, `run`; the views `view_name`/`view_prompt`,
  preset `load_views`/`save_views`) under `ui.NewPhotos`, opened by
  `RecordEditor._new_photos`. Blend anywhere (any two pictures, as a job of the
  queue, kept in History): `blend.submit`, `blend.run_job`, `blend.record`,
  `blend_words`, under `ui.BlendWindow`, opened by `ImageStudio.blend`. The view cube Angles asks on:
  `apps/image_studio/viewcube.py` (`cells`, `basis`, `facing`, `ViewCube`).
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

## apps/phone/ (Studio Assist Phone.cmd)
`apps/phone/server.py` over `apps/phone/page.html` (the whole page: markup, style and
script). Start at `Phone` (`chat`, `generate`, `state`, `gallery`, `picture`,
`make_room`), `Access` (who is served, the `--lan` passcode), `Handler` (`admitted`,
`get`, `post`, `chat`), `plan`/`serve`/`main`. `python apps/phone/server.py --check`
says what it would serve and where. Tests: `tests/test_phone.py`. AGENTS.md "The phone".

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
  `OPENCODE_READ_CAP` = `apps/opencode/read_cap.js`, an OpenCode plugin always loaded:
  a read with no line range on a file over 400 lines gets 400 lines and a grep hint.
  `OPENCODE_REPO_MAP` = `apps/opencode/repomap.py`, MCP server "repo" always loaded:
  `repo_map` (`t_map`: `folder_map` / `file_map`, `outline_py` via ast) and `repo_find`
  (`t_find`), built from disk on each call. Tests: `tests/test_repomap.py`.
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
- **Trainer** - `apps/opencode/trainer_mcp.py`: an MCP server for *Claude Code*
  (`.mcp.json`, "opencode-trainer"), not the app. `t_tasks`/`trouble_in` (the latter now
  `core.lessons.trouble_in`), `t_task` (`transcript`), `t_diff`, `t_session`, `t_lessons`,
  `t_self_review` (`_recent_tasks` + `lessons.self_review`: the same trouble recurring
  across the last N tasks - a suggestion only, never written), `t_keep`/`t_forget`
  (+ `publish` to `lessons.md`). The window sees its lessons through `Notebook._sync`.
  Tests: `tests/test_trainer_mcp.py`.
- **Tests** - `tests/test_opencode.py` (`FakeOpenCode` plays a server),
  `tests/test_codeaddons.py`, `TestElicitation` in `tests/test_mcp.py`.

## Tests and ledgers
`tests/test_<module>.py` per module (named after the old flat module name);
`python -m unittest tests.test_<module>`.
`tests/VERIFIED.md` - what already passed (don't re-test); `tests/LIVE_ACCEPTANCE.md`.
