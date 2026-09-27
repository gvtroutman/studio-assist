# Studio Assist - brief for OpenCode

You are editing Studio Assist, a Windows desktop app (Python + Tkinter) that drives
creative apps (After Effects, Premiere, Photoshop, Illustrator, Resolve, ComfyUI) with
a local LLM, one tab per app. The full guide is `AGENTS.md` (very long - do NOT read it
whole; use grep on it for the one topic you need). This page is what you must know.

## Where things live
`docs/CODEMAP.md` maps every module and the functions to start at - read it before
opening big files (`core/chat.py`, `core/agent.py`, `apps/image_studio/imagegen.py` are 3-7k lines;
read the one function you need, not the file).
- `core/agent.py` - engine: app registry (`APPS`, `AppSpec`), LLM client, MCP client,
  each app's `system_prompt`.
- `core/chat.py` - the Tkinter window. `core/ui.py`, `studio_*_ui.py` - other windows.
- `studio_*_mcp.py` - our MCP bridges (one per app); `core/mcp.py` - the MCP harness.
- `core/tasks.py` - task execution; `core/lessons.py` - what the model learns.
- `tests/test_<module>.py` - unit tests, one file per module.

## Rules that break things silently
- **Stdlib only.** Never add a dependency or `pip install`. No `requests`, `openai`, `mcp`.
- **Never name a Tk widget method `_w`, `_bind` or `quit`** - they shadow `tkinter.Misc`
  and the window never opens.
- **Nothing in `Chat._drain` / `_handle` / `_report` may raise**; errors go to `_report`,
  never to the screen as a traceback.
- **Do not remove the startup warm-up** request in `core/chat.py`; it looks redundant
  and is not.
- **Do not shorten tool descriptions** or touch `sanitize_schema()` without a test.
- Sizes in pixels go through `Chat._px()`.

## Tests
`python -m unittest tests.test_<module> -v` for the module you changed (whole suite:
`python -m unittest discover -s tests`; no network or apps needed). The general rules
- your copy, small edits, reporting - are in the other brief.
