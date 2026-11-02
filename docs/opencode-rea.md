# OpenCode improvements from REA inspection

2026-10-14. REA 6.4.0 inspected selected, unchanged JavaScript members from the
installed Codex and Claude Desktop packages. The implementation below transfers
task-state and recovery patterns into Studio Assist's OpenCode integration.
It does not train the local model or establish equivalence with either product.

## Evidence and scope

| Artifact | Original ASAR SHA-256 | Focused REA Evidence |
| --- | --- | --- |
| Codex 26.1007.2314.0 | `c5fdef328cfa2ad9095786f9c3a7aaa447e68834d31299d6ef2592da02029038` | `ev_99a5f9c07290f583352833be46b48abf0d8df2ed5b19664d27c228d28407aa90` |
| Claude Desktop 2.31226.1.0 | `efa9e2310cf943b333b153c093babb8b5a92f80a23cb68764dba9b29a0801d4a` | `ev_495f12ad5986ab79420ac7fd982463179ec2cf073c4c2179023d1509fd5432b6` |

Codex: three source files, 21,220 bytes, 5,805 AST nodes, 69 findings, no parse
failures. Application graph coverage is complete for this subset; semantic
coverage remains partial/unknown. Claude: three source files and its package
descriptor, 579,139 bytes, 145,349 AST nodes, 197 findings, no parse failures.
Claude's graphs are partial: the semantic expansion of `index.chunk-CrdXqIcE.js`
exceeded the available result budget, while static/module facts were retained.
Imports into unselected modules remain unresolved. These are static observations,
not evidence that a path executed. Native Codex/Claude Code prompts, their model
weights, and their full cognitive engines were outside this inspection.

The local investigation records are in `.work/rea-opencode/{codex,claude}/`:
`focused-evidence.json`, `focused-provenance.json`, and `findings.md`. The Claude
folder also has `focused-selected-facts.json`, with exact REA function IDs and
source ranges. They remain ignored rather than shipping proprietary source or
large analysis records in this repository. The original full-package attempts
hit an integrity mismatch and Windows access restrictions; the focused analysis
then ran successfully with approved access and bounded worker deadlines.

## Observations and design transfers

| Observed implementation | OpenCode design transfer |
| --- | --- |
| Codex's `local-conversation-plan-model-4b1858ff9c86.js` selects plan text from a completed turn and keeps its turn/item identity. | Recover requests and todo statuses for the exact OpenCode session, rather than a global last-task cache. |
| Codex's `local-conversation-thread-turn-entries-a91ee5eaba74.js` distinguishes retry errors, terminal failure, diffs and in-progress todo activity. | Keep edits, checks and unfinished work separate; reaching the repair limit cannot mean “done.” |
| Claude's `index.chunk-DY2pcWC0.js` includes a SDK tool runner with scheduled/in-flight compaction, pending-tool guards and an empty-summary fallback. | Carry unresolved tool calls through compaction, avoid inferring their success, and preserve OpenCode's existing compaction prompt. Failed fact retrieval adds an explicit omission notice. |
| Claude's coordinator/replay modules (`index.chunk-CrdXqIcE.js`, `index.chunk-Boh_xviB.js`) scope resumes to sessions, deduplicate work and distinguish exact replay from gaps. | Read persisted session facts after a restart, state omissions, and avoid blindly repeating writes with unknown outcomes. |

The table's right column is our design inference. The Claude tool runner is
packaged SDK code; its presence does not prove that native Claude Code uses that
compaction path. REA's authenticated syntax locations were checked against the
inert source when interpreting these patterns. No app code was executed.

## What changed

- `apps/opencode/task_context.js` is loaded with the existing read-cap plugin.
  Before compaction, it reads OpenCode's saved messages and todos through the
  scoped SDK. It adds the first/recent user requests, corrections, todo statuses,
  changed files, recent shell exits/errors and unresolved tool calls. Facts are
  quoted, omissions are named, and checks before a later edit are marked stale.
  The added context is bounded to about 10,000 characters; SDK reads time out
  after four seconds. It preserves the default prompt and appends to an existing
  custom compaction prompt. It makes no inference requests or project writes.
- `apps/opencode/mcp.py` reports `needs_input` when failing tests or open todos
  remain after the automatic repair budget. `opencode_wait` preserves that
  result, including when Agentic is turned off. A new instruction resets the
  exhausted budget. Permissions, merge approval and isolation stay in force.
- `docs/OPENCODE_ANY.md` keeps the general brief below 4,000 characters while
  adding observable success criteria, final-diff review, accurate verification,
  reuse of evidence and continuation from the unfinished step.

## Validation and use

`tests/opencode_task_context.mjs` exercises SDK-shaped saved sessions, original
constraints/corrections, task isolation, pending tools, stale checks, nonzero
exits, missing/cleared output, bounded omissions, restarted plugin instances,
custom compaction prompts, failed SDK calls and aborted timeouts. Python's
`tests/test_opencode_context.py` includes it in the normal suite. The bridge
tests exercise exhausted repair limits, repeated waits and a new instruction.
Registry tests verify both built-in plugins are loaded alongside add-ons.

The focused validation command is:

```text
python tests/offscreen.py tests.test_agent.TestAppRegistry tests.test_opencode tests.test_opencode_context tests.test_repomap tests.test_codeaddons.TestCodeAddons -q
```

109 tests passed. This verifies the implementation with recorded/fake contracts;
no live local-model quality benchmark or live inference run was performed.
Reopen Studio Assist/OpenCode to load the updated bridge, plugin and brief.

## Model choice and retry state from the Fable client

The follow-up investigation of the same installed Claude Desktop package traced
model selection, effort settings and refusal-event consumers. Exact original
declaration bytes were kept in order in a derived 24,265-byte source subset. REA
6.4.0 returned `ev_cf8fee75e0e544a2853f1b62bb7de7940ff8fa883b0da3f60e27073746b5084b`,
with 4,672 semantic nodes and 4,495 static relations. Its application coverage is
complete for that derived subset; semantic status remains unknown. Omitted
definitions and runtime behavior remain unresolved. The larger original-member
pass retained partial evidence after reaching its resource limit.

The relevant observations are the configured-model validation and the `etn`
configuration guard, plus SDK refusal-event handling that restores a previous
model only while the fallback still owns the selection. Eight REA function nodes
were mapped back to the original module's SHA-256 and source ranges; all 51
inspection anchors and nine extracted declaration segments passed independent
byte checks. The ignored `.work/rea-fable/` folder retains `deep-findings.json`,
`deep-verification.json`, provenance and complete REA evidence. These are static
client observations; they do not reveal Fable's weights or server decision code.

The transfer into Studio Assist uses its existing local-model controls:

- An unserved model pin now follows the app's automatic policy, including the
  best coding model for OpenCode, and explains the fallback. Its saved preference
  is retained; the chip distinguishes the requested pin from the actual client.
- A menu choice is checked again before saving. Busy, closed or replaced tabs
  cannot change that selection. A model switch warms its candidate strictly and
  restores the old client and context if it fails, conditional on candidate
  ownership; conversation and task records stay intact.
- HTTP and streamed draft refusals retry the same request once without the draft.
  State changes wait for a valid response; failed retries retain the draft. A
  partial text/tool stream is never replayed, an HTTP retry cannot trigger a third
  streamed attempt, and a late refusal cannot disable a newer model/draft pair.

This is original implementation of validation and conditional recovery patterns.
Inference continues on the configured remote host. No unsupported effort fields,
classifier rules or proprietary source were introduced.

`tests/test_model_recovery.py` exercises request preservation, failed and partial
retries, typed context errors, changing settings during a request, stale menu
choices, switch rollback, task preservation and startup versus switch warm-up.
The tests use fake inference responses; no live model-quality improvement is
claimed.

Validation: the 429-check inference, GUI, executor, activity and idle-model suite
passed on the hidden test desktop. A final same-model client handoff regression
then passed with the 65 focused model-selection, draft-retry and registry checks.
`git diff --check` passed. The broader suite's legacy GUI imports require the
tests directory on `PYTHONPATH` when invoked with package-qualified names:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'tests'
python -B tests/offscreen.py tests.test_agent tests.test_model_recovery tests.test_tasks tests.test_activity tests.test_model_idle -q
```

The supported extension points were checked against
[OpenCode's plugin documentation](https://opencode.ai/docs/plugins/) and the
[installed version's hook contract](https://github.com/anomalyco/opencode/blob/v1.18.34/packages/plugin/src/index.ts).
REA's analysis approach is documented in its
[JavaScript artifact guide](https://github.com/morluto/rea/blob/main/docs/javascript-artifact-reconstruction.md).

The [Claude Code source follow-up](opencode-claude-code-rea.md) adds durable
verification results, richer compaction facts and recoverable tool outputs.
