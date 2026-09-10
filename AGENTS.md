# AGENTS.md

Working notes for anyone — human or agent — changing this project.

## What this is

A chat app that drives creative apps with a **local LLM**, one tab per app. Two
moving parts:

- **`studio_agent.py`** — the engine. The **app registry**, an MCP stdio client, an
  OpenAI-compatible LLM client (streaming and not), JSON-Schema sanitizing, and
  environment probes. Also a working CLI: `python studio_agent.py --app resolve
  "what's on the timeline"`, or bare for a REPL.
- **`studio_chat.py`** — the Tkinter GUI, and the way the app is actually used.
  Launched with no console via `Studio Assistant.cmd` and the Desktop / Start Menu
  shortcuts.

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
│   └ MCP stdio ─┐
│                ▼
│      davinci-resolve-mcp  ── in-process ──►  DaVinci Resolve
└────────────────────────────────────┘
```

Both bridges **must** run on this machine — one talks to a CEP panel on port 7777,
the other links against Resolve's scripting API in-process. Only inference is remote,
because the RTX 5090 here is reserved for AE/Resolve rendering and must not be
occupied by a resident model.

## The app registry

`eng.APPS` is the single description of everything drivable. An `AppSpec` carries how
to find the app (`exe_globs`), how to tell it is running (`probe`), how to start its
bridge (`command`, `args`), what to expose (`groups`, `default_groups`), and how to
talk about it (`system_prompt`, `examples`, badge colours). Adding an app is an entry,
not a code change; `tests/test_agent.py` walks the registry and fails on a half-filled
one.

Two things stay derived, never hand-maintained:

- `DRIVABLE` is built from `APPS`, so the sidebar cannot advertise more than the agent
  can actually do. **Only add an entry when a bridge really exists.**
- `probe` is a strategy *string* (`"port:7777"`, `"process:Resolve.exe"`) rather than a
  callable, so the registry stays data a test can walk.

### What a `system_prompt` has to carry

The model driving these apps is small and local. It knows After Effects and Resolve in
general; it does not know *this bridge* at all. The prompt is where the bridge's own
conventions live, so each app's covers, in this order:

- **How the project is shaped** — the object graph, and what addresses each object.
- **Units — the ones that fail silently.** AE is seconds, RGB 0..1, opacity 0..100,
  scale in percent, origin top-left. Resolve is frames and timecode, `track_index` from
  1, `item_index` from 0. A model left to guess these produces something that renders
  happily and is wrong.
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
not exposed.

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
calls. `MAX_TOOL_DESC_CHARS` is 4000 and `tests/test_agent.py` guards it.

**Never name a Tk widget method `_w`.** Tkinter's `Misc` uses `self._w` for the
widget's Tcl pathname. Shadowing it sends `__repr__` into infinite recursion and the
window never opens. `Misc._bind` and `Misc.quit` are the same kind of trap. Audit new
method names against `tkinter.Misc` / `tkinter.Tk` — a test does it for you.

**Tk pack order: fixed-size widgets first.** An expanding sibling packed *before* a
fixed one claims the leftover space and pushes it off the edge. This bug shipped
twice — once hiding the Send button, once hiding the whole composer until the window
was resized. In `_build()` the composer is packed first, then the tab strip, then the
expanding transcript stack; inside the composer the Send button precedes the input.
`tests/test_agent.py` asserts the invariant; keep it passing.

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

- **Queue events carry a session id**: `self.q.put((kind, sid, payload))`. A `sid` of
  `None` means "whatever tab the user is looking at" — startup failures on the shared
  inference host have no app of their own. `_handle` resolves it.
- **The header describes the active tab only.** Status, the Send button and the
  `Start <app>` button all come from `_apply_status()` reading the active session, so a
  background tab finishing work never rewrites the header you are looking at.
- **Boot is lazy and idempotent.** `_ensure()` fires on first `_select()`; `booting`
  guards re-entry and is cleared in a `finally` so a failed bridge can be retried by
  switching away and back.

## Running and testing

```bash
python -m unittest discover -s tests -v      # no network, no apps needed
python studio_agent.py --list-groups         # registry sanity, no bridge started
python studio_agent.py --app resolve --list-tools   # needs the Resolve venv
python studio_chat.py                        # the real app (console attached)
```

`tests/` never touches the network, the creative apps, or the model. Tests that would
need a display skip themselves when there isn't one; the GUI tests stub
`installed_apps` so tab behaviour doesn't depend on what this machine has.

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
  from 0. The system prompts say all of this; keep them saying it. A test asserts the
  AE half.
- **The Resolve prompt forbids `resolve_control` action `quit`.** The tool exists and
  works; closing the user's Resolve mid-session costs unsaved work. A test asserts the
  prohibition is still in the prompt.
- App marks are **drawn** two-letter badges, not extracted `.exe` icons — crisp at any
  DPI and no shipped assets. The shortcut icon follows the same rule: `make_icon.py`
  generates `studio-assistant.ico`, so the mark changes by editing colours rather than
  by opening a paint program.
- Errors must reach the user as prose. `_guard()` catches everything off the UI thread;
  tracebacks go to `studio_assistant_error.log`, never to the screen.
