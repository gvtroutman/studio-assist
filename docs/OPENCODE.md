# Studio Assist - brief for OpenCode

You are editing Studio Assist, a Windows desktop app (Python + Tkinter) that drives
creative apps (After Effects, Premiere, Photoshop, Illustrator, Resolve, ComfyUI) with
a local LLM, one tab per app. The full guide is `AGENTS.md` (very long - do NOT read it
whole; use grep on it for the one topic you need). This page is what you must know.

## Where things live
`docs/CODEMAP.md` maps every module and the functions to start at - read it first.
- `core/agent.py` - engine: app registry (`APPS`, `AppSpec`), LLM client, MCP client,
  each app's `system_prompt`.
- `core/chat.py` - the Tkinter window; `core/ui.py` - shared widgets.
- `core/mcp.py` - the MCP harness; `core/tasks.py` - task execution;
  `core/lessons.py` - what the model learns.
- `apps/<app>/` - one package per tab: its bridge (`mcp.py`, or `apps/adobe/<app>.py`)
  and its windows (`ui.py`, `view.py`...).
- `tests/test_<module>.py` - unit tests; `tests/test_<app>.py` for an app package.
- There are no `studio_*.py` modules any more except the launcher `studio_chat.py`
  and `studio_update.py`; do not look for the old names.

## Finding code fast
`core/chat.py`, `core/agent.py` and `apps/image_studio/imagegen.py` are 3-7k lines.
Never read a big file whole - a read with no line range shows only its first 400 lines.
1. Grep for the definition: `def _build_composer`, `class Chat`. Grep for one exact
   word from the task, not a loose pattern like `theme|Theme` over the whole repo.
2. Read 60-150 lines with offset and limit around the match.
3. Grep for callers only if you must change a signature.

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
