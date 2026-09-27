# Studio Assist - brief for OpenCode

You are editing Studio Assist, a Windows desktop app (Python + Tkinter) that drives
creative apps (After Effects, Premiere, Photoshop, Illustrator, Resolve, ComfyUI) with
a local LLM, one tab per app. The full guide is `AGENTS.md` (very long - do NOT read it
whole; use grep on it for the one topic you need). This page is what you must know.

## Where things live
`docs/CODEMAP.md` maps every module and the functions to start at - read it before
opening big files (`studio_chat.py`, `studio_agent.py`, `studio_imagegen.py` are 3-7k lines;
read the one function you need, not the file).
- `studio_agent.py` - engine: app registry (`APPS`, `AppSpec`), LLM client, MCP client,
  each app's `system_prompt`.
- `studio_chat.py` - the Tkinter window. `studio_ui.py`, `studio_*_ui.py` - other windows.
- `studio_*_mcp.py` - our MCP bridges (one per app); `studio_mcp.py` - the MCP harness.
- `studio_tasks.py` - task execution; `studio_lessons.py` - what the model learns.
- `tests/test_<module>.py` - unit tests, one file per module.

## Rules that break things silently
- **Stdlib only.** Never add a dependency or `pip install`. No `requests`, `openai`, `mcp`.
- **Never name a Tk widget method `_w`, `_bind` or `quit`** - they shadow `tkinter.Misc`
  and the window never opens.
- **Nothing in `Chat._drain` / `_handle` / `_report` may raise**; errors go to `_report`,
  never to the screen as a traceback.
- **Do not remove the startup warm-up** request in `studio_chat.py`; it looks redundant
  and is not.
- **Do not shorten tool descriptions** or touch `sanitize_schema()` without a test.
- Sizes in pixels go through `Chat._px()`.

## How to work
1. Before editing, read the functions you will change and grep for their callers.
2. Make the smallest change that does the task. The user approves every edit by
   reading its diff, so several small edits beat one huge one.
3. Match the surrounding code's style and comment density.
4. Run the tests for the module you changed:
   `python -m unittest tests.test_<module> -v`
   (whole suite: `python -m unittest discover -s tests`; no network or apps needed).
5. When finished, say in a few lines which files and functions you changed and what the
   tests reported. If you could not finish, say exactly what is left.

Keep a todo list for tasks with more than two steps, and tick items off as you go -
your memory of earlier steps may be compacted away, the todo list is not.
