# Code map

Where things are, so you open one file at the right function instead of reading
around. Symbols, not line numbers: grep `def name` / `class Name` to land on one.
Rules and the why behind them are in `AGENTS.md` (long - grep it, don't read it).
Update this page when you add, split or rename a module.

## Core (every tab)
| File | Lines | What it is | Start at |
| --- | --- | --- | --- |
| `studio_chat.py` | 5.3k | The Tkinter window: tabs, chat, approval cards | `Chat`, `_drain`/`_handle`/`_report`, `_elicit`/`_show_elicit`/`_diff_box`/`_settle_elicit` |
| `studio_agent.py` | 3.7k | Engine: app registry, LLM client, MCP client, prompts, CLI | `APPS`, `AppSpec`, `ServerSpec`, `LLM`, `MCPClient`, `Router`, `*_PROMPT` |
| `studio_mcp.py` | 1.2k | MCP harness every bridge is built on | `Server`, `tools_from_table`, `elicit`, `Declined`, `progress`, `cancelled`, `Loopback` |
| `studio_tasks.py` | 0.9k | Task execution and recoverable task records | |
| `studio_lessons.py` | 0.3k | What the model learns after each task, per app | |
| `studio_toolsmith.py` | 0.4k | Tools the model writes for itself from its own bridge tools | |
| `studio_ui.py` | 0.3k | Palette roles and drawing primitives | |
| `studio_procs.py` | 0.3k | Child processes that die with the parent | |
| `studio_files.py` | 0.1k | Attachments described for a model | |
| `studio_doctor.py`, `studio_update.py` | | Where things are kept / health; sync with GitHub | |

## Bridges (one MCP stdio server per app)
`studio_premiere_mcp.py`, `studio_photoshop_mcp.py`, `studio_illustrator_mcp.py`,
`studio_comfy_mcp.py`, `studio_research_mcp.py` (Chat tab: files + web, read-only),
`studio_opencode_mcp.py`. Adobe transport: `studio_com.py` (COM) and `studio_cep.py`
(CEP panel, `premiere_panel/`). After Effects and Resolve use outside npm servers.

## Image Studio (ComfyUI)
`studio_images_ui.py` (tab, 5.9k) over `studio_imagegen.py` (engine, 6.7k).
Scene Builder: `studio_scene_ui.py` over `studio_scene.py`, `studio_mannequin.py`,
`studio_pose.py`. Add-ons: `studio_catalog.py`, `studio_civitai.py`, `studio_hub.py`,
`studio_addons.py`, `studio_discovery.py`, `studio_model_sources.py`. Identity:
`studio_lora_train.py`, `studio_facefusion.py`, `tools/`. Critic: `studio_critic.py`.
Node views: `studio_comfy_view.py`, `studio_nodes_ui.py`. Workflows: `comfy_workflows/`,
`recipes/`, custom nodes `comfy_nodes/`.

## Other tabs
Terminal: `studio_terminals_ui.py` + `studio_consoles.py`. Milanote: `studio_milanote.py`.
Icons: `studio_icons.py`, `make_icon.py`.

## OpenCode, end to end
```
Chat tab (studio_chat)  --model briefs-->  studio_opencode_mcp (bridge, stdio)
   ^  approval card                          |  HTTP + Basic auth (server.key)
   |  (_show_elicit)                         v
   +---- elicitation/create <---- settle() <-- opencode serve (ServerSpec.launch)
                                              |  edits the workspace (this repo)
                                              +-> LM Studio on the LLM PC
```
- **Start/stop, config** - `studio_agent.py`: `ServerSpec` (`launch`, `stop`,
  `write_config`, `fit_window`, `model_for`), `opencode_config`, `OPENCODE_PERMISSIONS`,
  `OPENCODE_BRIEF` (= `docs/OPENCODE.md`), `own_repo`, `opencode_exe`,
  `OPENCODE_GROUPS`, the `id="opencode"` entry in `APPS`, `OPENCODE_PROMPT`.
  State (config, password, `last_session`, log) in `%LOCALAPPDATA%\StudioAssistant\opencode`.
- **Bridge** - `studio_opencode_mcp.py`: `t_ask` -> `start_task` (git worktree per
  task, `tasks.json`) -> `prompt` -> `finish` -> `follow` (woken by `Events`, the
  `/event` stream) -> `settle` (permissions + questions of the session `Family`;
  `granted`/`add_grant` keep "always") -> `ask_permission` / `ask_question` ->
  `studio_mcp.elicit`; `report` + `context_report` summarise; `after_ask` runs
  `tests_for` and makes a `checkpoint`. `t_merge` / `t_undo` / `t_discard` ask through
  `confirm`. Every session call passes `task_dir(sid)` as `?directory=`. `ROUTES` is
  every server route used.
- **Approval UI** - `studio_chat.py`: `_elicit`, `_show_elicit`, `_diff_box`, `_settle_elicit`.
  Direct mode: `_toggle_direct`, `_direct_turn`.
- **Add-ons** - `studio_codeaddons.py` (records, `config`, MCP registry / npm / skills
  search and install) and `studio_codeaddons_ui.py` (`AddonsWindow`).
- **Tests** - `tests/test_opencode.py` (`FakeOpenCode` plays a server),
  `tests/test_codeaddons.py`, `TestElicitation` in `tests/test_mcp.py`.

## Tests and ledgers
`tests/test_<module>.py` per module; `python -m unittest tests.test_<module>`.
`tests/VERIFIED.md` - what already passed (don't re-test); `tests/LIVE_ACCEPTANCE.md`.
