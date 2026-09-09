# AGENTS.md

Working notes for anyone — human or agent — changing this project.

## What this is

A chat app that drives **Adobe After Effects** with a **local LLM**. Two moving parts:

- **`ae_agent.py`** — the engine. MCP stdio client, OpenAI-compatible LLM client
  (streaming and not), JSON-Schema sanitizing, environment probes. Also a working
  CLI: `python ae_agent.py "make a title card"`, or bare for a REPL.
- **`ae_chat.py`** — the Tkinter GUI, and the way the app is actually used.
  Launched with no console via `AE Agent.cmd` and the Desktop / Start Menu shortcuts.

### Where the work happens

Inference is **remote**; tools are **local**. That split is not negotiable:

```
this PC (Adobe workstation)                    tailnet peer
┌──────────────────────────────────┐          ┌──────────────────────┐
│ ae_chat.py / ae_agent.py         │  HTTP    │ LM Studio            │
│   └ MCP stdio ─┐                 │ ───────► │ 100.127.17.38:1234   │
│                ▼                 │          │ OpenAI-compatible    │
│  @engine-room/after-effects-mcp  │          └──────────────────────┘
│                │ ws 127.0.0.1:7777
│                ▼
│  CEP panel inside After Effects
└──────────────────────────────────┘
```

The bridge **must** run on this machine — it talks to the CEP panel on port 7777.
Only inference is remote, because the RTX 5090 here is reserved for AE/Resolve
rendering and must not be occupied by a resident model.

## Hard-won constraints — read before editing

**Stdlib only.** No `openai`, no `mcp`, no `requests`, no pip step. This runs on an
Adobe workstation where a broken Python environment costs real production time.
Tkinter is the GUI for the same reason. Do not add dependencies.

**`sanitize_schema()` is load-bearing.** The AE bridge describes fixed-length arrays
(`bgColor`, `position`) in JSON Schema 2020-12 tuple form: `prefixItems` next to
`"items": false`. LM Studio's schema-to-grammar converter is draft-07 shaped and
rejects the boolean with `Unrecognized schema: false` — **HTTP 400 for the entire
request**, not just that tool. It hit 13 of 45 tools. Any other OpenAI-compatible
runner (llama.cpp, vLLM grammar mode) will hit the same wall.

**Never name a Tk widget method `_w`.** Tkinter's `Misc` uses `self._w` for the
widget's Tcl pathname. Shadowing it sends `__repr__` into infinite recursion and the
window never opens. Audit new method names against `tkinter.Misc` / `tkinter.Tk`.

**Tk pack order: fixed-size widgets first.** An expanding sibling packed *before* a
fixed one claims the leftover space and pushes it off the edge. This bug shipped
twice — once hiding the Send button, once hiding the whole composer until the window
was resized. The composer is packed before the transcript, and the Send button before
the input, on purpose. `tests/test_agent.py` asserts the invariant; keep it passing.

**Do not remove the startup warm-up.** It looks like a redundant throwaway request.
It is not: ~14k tokens of tool schemas take about a minute to prefill cold. The
warm-up pays that at launch against *the exact prompt prefix a real message uses*, so
the first question returns in seconds. LM Studio's prefix cache survives across
processes, which is why this works at all.

**Never test the GUI with synthetic keystrokes.** `SendKeys` types into whatever
window has focus, not the one you meant. It has already leaked a test sentence into
the user's chat window mid-run. Test by constructing `Chat()` in-process and calling
its handlers, and assert on widget geometry for layout. Screen-capturing the window
to look at it is fine — that's read-only.

## Running and testing

```bash
python -m unittest discover -s tests -v   # no network, no AE needed
python ae_agent.py --list-tools           # needs npx; AE may be closed
python ae_chat.py                         # the real app (console attached)
```

`tests/` never touches the network, After Effects, or the model. Tests that would
need a display skip themselves when there isn't one.

To exercise the live path you need After Effects **open** — the bridge binds 7777
only while AE runs. `check_setup` (via the MCP server) diagnoses the whole chain and
its `nextSteps` are reliable; relay them rather than guessing.

## Conventions

- Tool groups in `eng.GROUPS` keep the exposed tool count down; a 3B-active MoE gets
  sloppy shown all 76 at once. `DEFAULT_GROUPS` is the working set.
- Identify AE layers by `id`, never `index` — an index shifts on every insert. The
  system prompt says so; keep it saying so.
- `eng.DRIVABLE` lists apps with a real bridge. Only After Effects qualifies today.
  Extend it only when a bridge actually exists, so the UI stays honest.
- App marks in the sidebar are **drawn** two-letter badges, not extracted `.exe`
  icons — crisp at any DPI and no shipped assets.
- Errors must reach the user as prose. `_guard()` catches everything off the UI
  thread; tracebacks go to `ae_agent_error.log`, never to the screen.
