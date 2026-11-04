# OpenCode improvements from Claude Code source inspection

2026-10-14. This follows the [earlier Desktop-package investigation](opencode-rea.md)
using a public mirror labelled Claude Code 2.1.88, the source-map leak reference
requested by the user. These are Studio Assist changes around its installed
OpenCode; inference still uses the local model on the remote inference host.

## Reference and evidence

The [reference mirror](https://github.com/shipinlei/claude-code-source-code/tree/2ca5ddabfed5f220812ea11f029eda03b21bc4c1)
is pinned to commit `2ca5ddabfed5f220812ea11f029eda03b21bc4c1`. Its attribution was
not authenticated against the original npm source-map bytes. Four unchanged
files were downloaded and analyzed as inert text; the reference was never run.
No reference source or prompt is shipped in this repository.

REA 6.4.0 returned
`ev_128b2d1c25d43137172af1a2fb1138d399cf4a60006c51cb835e2112ff5aaf70`.
The selected-directory SHA-256 is
`06cc9dceb388433eb250ca527ccb0a425e53d97402e793e0e2ed0eeb845834a6`.
It read 86,782 bytes across four TypeScript/TSX files, visited 5,088 AST nodes
and retained 147 findings, with no parse failures. Application graph coverage
is complete for this subset. Semantic coverage is unknown, with partial
relationship families and unresolved imports into unselected modules. Static
evidence does not establish execution, whole-product behavior or model quality.

Complete evidence, file URLs/digests and selected function identities are kept
in ignored `.work/rea-claude-code/{evidence,provenance,findings}.json`.

## Observations and transfers

| Mirrored source observation | Studio Assist design transfer |
| --- | --- |
| [`autoCompactIfNeeded`](https://github.com/shipinlei/claude-code-source-code/blob/2ca5ddabfed5f220812ea11f029eda03b21bc4c1/src/services/compact/autoCompact.ts#L241) carries consecutive failure state, stops attempts at its limit and resets after success. REA function `jsrg_node_8e8b9289090db0639a70101ac55514208647750a4dcc8cbf3c5668a6aeaef5b2`, lines 241–351. | An idle session or saved checkpoint cannot clear a failed/unavailable check. Retain verification state and the existing repair bound across retries and waits. |
| [`getCompactPrompt`](https://github.com/shipinlei/claude-code-source-code/blob/2ca5ddabfed5f220812ea11f029eda03b21bc4c1/src/services/compact/prompt.ts#L293) selects a prompt covering technical decisions, errors, pending tasks and current work. REA function `jsrg_node_bb661d6126c8c441bfd051883d6a0e14591876d9e391ba807109bf92f11e2793`, lines 293–303; prompt text at lines 61–77. | Recover recent assistant findings/decisions alongside recorded requests, tools and todos; keep these quoted claims subject to verification. |
| [`maybePersistLargeToolResult`](https://github.com/shipinlei/claude-code-source-code/blob/2ca5ddabfed5f220812ea11f029eda03b21bc4c1/src/utils/toolResultStorage.ts#L272) replaces oversized content with a reference to retained content; empty results have explicit markers. REA function `jsrg_node_d26bc8c16b7029f7eb4302b7254ccf98e6dfac388f3b027700dc80357276481b`, lines 272–334. | Report bounded command/output facts and provide a tool to recover the original saved result in windows. OpenCode already persists those facts, so no second output store is needed. |

The right column describes our design choices, not copied implementations.

## Resulting behavior

- Verification timeouts, test startup failures and errors inspecting the task's
  copy now leave work unresolved. Agentic retries stay bounded; with Agentic
  off, the bridge reports `needs_input` without claiming a repair-limit event.
- `after_ask` saves the checked modules and their result on the task. Until a
  failed/unavailable run actually passes, those modules are included in later
  verification even if the prior checkpoint leaves no fresh diff or a later
  edit has no matching test. Repeated waits preserve the unresolved verdict.
  A new instruction can recheck it. Timed-out tests retain their diagnostic tail.
  Test processes run in the project's kill-on-close job objects; completion,
  Stop and timeout release their descendants too.
- Compaction recovers the latest three assistant text records, excludes summary,
  reasoning, synthetic/ignored text and other sessions, bounds each record and
  discloses omissions. Their claims remain separate from recorded tool exits.
- Session reports include command exits and bounded output. `opencode_tool_result`
  reads one saved call's original output/error in windows, scoped by session and
  call ID. Cleared, missing and empty outputs are distinguished. Shortened
  reports retain their ending and tell the model how to recover omitted facts.

Tests use recorded/fake OpenCode contracts and temporary repositories; no live
local-model quality benchmark was performed. Permissions and merging remain
under the user's existing controls. Reopen Studio Assist/OpenCode to load the
updated bridge and compaction plugin.

Validation: 266 focused registry, add-on, OpenCode, verification, result,
compaction, MCP, executor and process tests passed. This includes a real timed-out
test that starts a descendant process, checks the retained diagnostic and confirms
that descendant exited. The JavaScript compaction checks run through their Python
wrapper in this suite. `git diff --check` passed.

```text
python -B -m unittest tests.test_agent.TestAppRegistry tests.test_codeaddons.TestCodeAddons tests.test_opencode tests.test_opencode_verification tests.test_opencode_results tests.test_opencode_context tests.test_mcp tests.test_tasks tests.test_procs -q
```
