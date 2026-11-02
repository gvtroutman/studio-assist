# Live tool activity from Claude Desktop inspection

2026-10-14. The requested reference is Claude Desktop 2.31226.1. REA 6.4.0
analyzed unchanged members extracted from its installed ASAR. Its original
SHA-256 is `efa9e2310cf943b333b153c093babb8b5a92f80a23cb68764dba9b29a0801d4a`.

The useful evidence is
`ev_1a2e36ca85eeefbb235ebffa48b694f2ae96a7d90ca9749448406b115ea335bb`:
`index.chunk-D7ye_MQA.js` plus the original package descriptor. REA parsed
1,990,968 bytes, visited 486,277 AST nodes and retained 295 findings without
parse failures. Semantic coverage is partial: the graph reached its 100,000-node
ceiling. Imports outside this subset are unresolved. This is static evidence,
not an observation of Claude running or proof of the remotely delivered UI's
rendering behavior.

REA identifies `wr`, function node
`jsrg_node_ed505787e8cd49bd3a19e83b6f3238112c393bab3044a9df4399fa7328a643c4`,
at line 1, columns 52009–52761. The member's SHA-256 is
`da9c4c66914e47630cca65a35a6cfd7ea29d3c6cad688b47c407cae627206462`.
Reading those authenticated syntax locations shows a map of active tool IDs:
assistant tool-use records add IDs and names; corresponding tool results remove
them; progress references keep unresolved IDs active. It starts after the latest
top-level terminal result and distinguishes nested records. That state-management
pattern is observed; the Studio Assist strip's layout is our design choice.

An initial bounded selection of five smaller modules and the descriptor produced
`ev_e146f393fbc5ad14e536d2a5a4b6ecdb429ca2a9c23a949c525702de870db139`.
It established separate text/tool record handling and session/event identities,
but did not answer the active-tool tracking question, which motivated the second
selection. Complete evidence, provenance and selected facts remain in the ignored
`.work/rea-tool-activity/` folder; proprietary source is not shipped.

Studio Assist now displays a per-tab activity strip while a reply is being made,
with expandable arguments, progress and results. Preparing a streamed tool call
is distinct from executing it. Text streaming leaves tool history visible;
interruption leaves unmatched operations with unknown outcomes. History is bounded
to 80 displayed calls and 2,000 characters per detail, with explicit omissions.

OpenCode tool updates use its documented
[part/state contract](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/schema/src/v1/session.ts).
The bridge carries these as optional MCP progress metadata, deduplicated by
session/call ID. Polling recovers root-session activity if SSE drops; the final
saved messages settle root results. Existing permission and completion logic
continues to use its own authoritative routes.

Offline tests exercise preparation before dispatch, cancellation during partial
arguments, stable IDs, same-name nested calls, interruption, bounded omissions,
expired progress tokens, display callback failure, SSE/poll equivalence, tab
isolation, live text and minimum-window/theme layout. No live inference or Claude
UI equivalence benchmark is claimed. See REA's
[artifact workflow](https://github.com/morluto/rea/blob/main/docs/javascript-artifact-reconstruction.md)
for the analysis boundary.

Validation: 211 activity/executor/MCP/OpenCode tests and 303 inference/GUI tests
passed. The final 13 activity tests and three focused GUI tests passed again
after the display refinements. Dark and minimum-size light previews were rendered
on the hidden test desktop and inspected. Commands:

```text
python tests/offscreen.py tests.test_activity tests.test_tasks tests.test_mcp tests.test_opencode -q
python tests/offscreen.py discover -s tests -p test_agent.py -q
```

The subsequent UI revision replaces raw transcript headers with compact tool
cards: readable action names, drawn icons, explicit outcome badges, a separate
context line and reported progress counters. Exact calls and results remain in
selectable folded transcript text. The cards follow live font/theme/rounding
preferences and fit the transcript width. Interrupted calls stop animating and
remain unconfirmed; completed calls drop their live timing badge. The pinned
activity strip uses the same readable labels. This layout is a Studio Assist
design change, rather than an observed reproduction of Claude's UI. The revised
UI passed 307 GUI/inference tests and 15 activity tests; the user's five-call
OpenCode example was rendered and inspected in both themes and at minimum size.
