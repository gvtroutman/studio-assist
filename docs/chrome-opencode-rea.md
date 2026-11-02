# Chrome control for OpenCode and the local LLM

2026-10-14. Studio Assist's OpenCode add-on can attach to existing Chrome tabs
through Microsoft's Playwright extension. The local model receives accessibility
snapshots, finds controls and sends clicks, text and form input. Browser tools run
on this PC; inference stays on the configured remote LM Studio host.

## Setup and use

1. Install the official [Playwright extension](https://chromewebstore.google.com/detail/playwright-extension/mmlmfjhmonkocbjadbfplnigmagldckm)
   in the Chrome profile where the relevant site is already open.
2. In Studio Assist, open **OpenCode → Add-ons → MCP servers → Installed →
   Connect Chrome…** and select **Add Chrome tools**. Restart OpenCode with the
   Add-ons window's button when its current task has finished. A registered
   add-on takes effect on the next normal OpenCode start as well.
3. Ask OpenCode, preferably in Direct mode: “Read the Docker container's status
   in the Chrome tab I connect. Report what is running; do not change anything.”
4. The first browser tool opens Playwright's connection UI. Choose the tab to
   connect. Browser access is established at that point; an MCP status reading
   “Connected” alone only establishes the tool server connection.

The official [extension guide](https://github.com/microsoft/playwright/blob/main/packages/extension/README.md)
describes tab selection and separate groups for connected clients. Give OpenCode
its own tab if Claude is controlling another one. Disconnect in the extension,
or turn off the Chrome add-on and restart OpenCode. The preset leaves connection
approval enabled and does not store an extension token.

The preset pins `@playwright/mcp` **0.0.83**. It uses a matching installation in
`.runtime/browser/node_modules/@playwright/mcp` with Node on PATH when present;
otherwise OpenCode runs `npx -y @playwright/mcp@0.0.83`. That is a separate npm MCP
tool, with no new Python or project package dependencies. The package's own
Playwright version is `1.64.0-alpha-1790635538000`, as returned by MCP initialize.

The server is stdio and uses `--extension --no-webmcp --image-responses omit
--codegen none`, with 10-second action and 30-second navigation limits. The
record excludes 11 tools from the 25-tool verified contract, leaving 14 for
ordinary page interaction. Arbitrary code, network bodies, uploads and screenshots
are excluded. DOM/accessibility control works for a text-only model; pixel-only
canvas controls require a separate vision-capable integration.

Every remaining Chrome tool retains `chrome_*: ask`. Agentic continues to cover
only permitted worktree edits/commands. Browser actions affect the live site;
worktree checkpoints, undo and merge cannot reverse them. Task-scoped grants
remain under the user's existing controls. Recorded browser interaction suppresses
the coding agent's “nothing changed in your copy” nudge, so an external-site task
does not demand an unrelated repository edit. An attempt is never independent
proof of success; the model's browser brief requires reading the resulting state.

The browser brief is appended to existing project and lesson instructions only
while this add-on is enabled. Existing records with a different server named
`chrome` are preserved and the setup explains the collision.

## REA evidence

The latest installed Claude Chrome extension package on disk was **1.0.100**,
extension ID `fcoeoabgfenejglbffodgkkbkcdhcgfn`. This is not a measurement of which
extension version the running Chrome window had loaded. Four unchanged members
were copied to ignored `.work/rea-chrome/reference/` and inspected as inert text.
Browser history, cookies and credentials were not read; reference code was not run
or incorporated into Studio Assist.

REA **6.4.0** returned evidence
`ev_af20dbafedf00742844bc1a59502ba69f8580c4ba5b031781da29685065661ef`,
root artifact digest
`3531e7c9fb87d282f23a7f6b2fc4016bddee8daff84a2cff43528e942d4f52bd`.
It read 709,037 bytes across the manifest and three scripts, parsed all three
JavaScript files, visited 176,539 AST nodes and retained 202 findings, with no
parse failures. Application coverage is partial. Semantic coverage is partial
and truncated under the 512 MiB/15-second worker bounds; omitted imports and
runtime execution remain unresolved. This does not reconstruct Claude's server
reasoning or establish equivalent model quality.

| Installed member | SHA-256 |
| --- | --- |
| `manifest.json` | `067a9b991eab4f35b81bafcd28caa80f261d1f9d64d4c75f2ae71c6304263012` |
| `accessibility-tree.js-B-oUarrX.js` | `5b179ad53bd86324859cd9fdb44bc10d8adba78ef6d9f7af0818988838b5a6b2` |
| `mcpPermissions-IBi8ZJfc.js` | `a4d977dd55640ecfafb90a96241a19cfeae50fd946739fb93900c4d90eb56993` |
| `service-worker.ts-DPwYJN6x.js` | `10a7fbdf0c2495939479a3edddd56e8473c172231f9aae2513825539ccce7835` |

An independent Babel pass recorded 195 syntax anchors with original-member hashes
and UTF-16 character ranges in `.work/rea-chrome/static-anchors.json`. Observed
syntax includes a debugger attachment, CDP mouse dispatch and screenshot capture
in `mcpPermissions`, native messaging in the worker, and page-reference recovery
in the accessibility script. For example, `dispatchMouseEvent` spans
191599–192785 and `screenshot` 198035–201432 in `mcpPermissions`; the native-host
probe spans 42973–44215 in the worker. The manifest limits external website
connections to Claude domains. A transport probe's presence does not establish
what executed in the live session.

Our design transfer is a user-selected browser connection, compact structured
observations and an observe/action/verify loop. We use Microsoft's public MCP
implementation and [documented existing-browser extension mode](https://playwright.dev/mcp/configuration/browser-extension),
with Studio Assist's own setup, permission configuration and instructions. It
does not repurpose Claude's extension or call Anthropic's API.

## Validation

The pinned package was installed separately with npm install scripts disabled.
Its real stdio MCP initialize/tools-list exchange succeeded without opening or
attaching to any Chrome tab. The recorded public contract is
`tests/fixtures/chrome-playwright.json`.

Offline tests cover the preset's permissions, retained briefs, disabled/removal
behavior, exact runtime version matching, tool subset, setup buttons, record
collisions and live-browser activity versus missing code edits. GUI tests run on
the project's hidden desktop. End-to-end local-model control of a connected
Chrome tab remains unmeasured until the user installs the extension and selects
the tab. No NAS changes were attempted for this integration task.

Validation: **269 tests passed** across add-ons, the app registry, OpenCode,
verification, saved results, compaction, MCP and the executor. After the final
setup status-refresh change, all **17 add-on tests passed** again.
`git diff --check` passed.

On this workstation the tested add-on was registered in OpenCode's existing
`addons.json`, preserving all 43 prior records. The original file has a
`addons.chrome-backup.<digest>.json` backup in the same state folder. The app was
not restarted; registration takes effect at the next normal OpenCode start.
The receipt is retained in ignored `.work/rea-chrome/registration-receipt.json`.
