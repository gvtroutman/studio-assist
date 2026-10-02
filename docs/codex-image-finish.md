# Codex image finishing

Enable **Codex finish (faces covered for handoff)** in Image Studio's
setup options. The existing generation, identity and repair passes run first.
The pipeline then reaches **Codex finish** and says **Waiting for Codex**.
The GPU lane is released; other jobs can continue.

Select the saved picture and choose **Image actions → Finish with Codex**, or
right-click the picture and choose **Finish with Codex**. Review the green boxes:
every face to keep must be fully inside a protected rectangle. Scene Builder's
head regions provide initial boxes; drag to add boxes and right-click to remove
one. Cover the whole face and its edges; cover hair too when you want it private.
Mark background faces and reflections too when they should stay private or unchanged.

Describe the finish and click **Copy handoff**. Paste into a Codex chat with
imagegen. Every protected rectangle is replaced by solid opaque grey before
export. The preview then shows the covered image supplied to Codex. The shared
folder `image-studio/codex-handoffs/<id>/` contains only `redacted.png` and
`request.md`, with a destination for `finished.png`. The request references only
that redacted image and omits the original scene/identity metadata and reference
photo paths. No API key is needed.

The original pixel snapshot and `handoff.json` stay separately in
`image-studio/codex-private/<id>/` for local restoration. Share only the redacted
image and request. The manifest and private folder are never included in the
copied request. This separation controls what the handoff supplies; it does not
restrict Codex's other filesystem permissions. Earlier handoffs and images already
shared are unchanged; create a fresh handoff to use face redaction.

After Codex saves the result, click **Import finished image**. A previously
exported `handoff.json` from the private folder can also be selected after reopening the app. Import
requires a supported 8-bit, non-interlaced PNG with the original canvas proportions.
Larger or smaller results with matching proportions are fitted to the source
resolution before face restoration; the original head pixels are never resized.
It copies the original RGBA pixels into every protected rectangle, then writes
a new PNG and a new History record. The original image remains available, and
the pipeline completes when all pictures in the batch have been imported.

Face protection covers the marked rectangles exactly. Review their coverage
and keep framing, head positions and nearby lighting unchanged during the edit;
an edit that moves a person can make restored regions visibly misalign.
If Codex changes the canvas proportions, ask it to keep the original framing before
importing. This feature is a manual Codex handoff, not a direct API connection.

The result retains the source settings for **Reuse**, and links back to the
source image and handoff. **Finish again** creates another handoff; a ComfyUI
seed cannot reproduce a Codex edit. Earlier ComfyUI graphs and pass claims are
not attributed to the imported image.
