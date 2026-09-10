# Creative-task acceptance checks

Run these manually on disposable projects after offline tests pass. They are not
part of unittest discovery and must never run against a production project by
default. Use the bridge/version and remote model intended for normal work; record
both along with the saved task path and actual output. Check results in the app,
not just in the assistant's final answer.

| Task | Measurable checks | Visual review |
| --- | --- | --- |
| AE: Create a 1920×1080, 5-second, 24 fps title reading “Safety First”; fade in over 12 frames. | Exact text; 1920×1080; 24 fps; duration 5; fade lasts 0.5 seconds; final opacity 100; read-back after last edit. | Centering, no clipping, adequate contrast; inspect start, mid-fade, full-opacity and end frames; play animation. |
| AE: Add a centered 300-pixel red circle. | Ellipse 300×300; geometry and visible fill exist; RGB [1,0,0]; centered relative to actual comp dimensions. | Circle visibly centered, no unexpected group/layer offset or stroke. |
| Resolve: Assemble three known test clips in a specified order on video track 1. | Correct source clips, order, source ranges, frame rate, track and total length; no unintended gaps or overlap. | Play every cut; check audio continuity and any scaling/cropping. |
| Resolve: Set up an H.264 job without starting it. | Codec/format discovered from this installation; requested path and settings; exactly one queued job; rendering has not started. | Inspect job settings in Deliver. |

Also exercise these recovery scenarios with disposable work:

1. Stop during a multi-step edit. Confirm later calls are skipped and finished edits
   are accurately described; Stop does not promise rollback.
2. Close and immediately reopen the same app tab during a reply. No old tokens,
   previews or ready/idle events should enter the new tab.
3. Interrupt a bridge after sending a write. Resume the saved task, inspect the
   project, and confirm the unknown call is never blindly repeated.
4. Continue a task long enough to exceed the inference history budget. Exact copy,
   dimensions and recorded object IDs must remain in the task record. Verify IDs
   against the current project before reusing them.
5. With visual critique disabled, the assistant must not claim it visually reviewed
   a screenshot. With it enabled, verify critique identifies real, actionable
   defects and does not claim a still image proves motion/audio quality.

Record pass/fail per criterion, user interventions, failed tool calls and total
task time. Compare models or prompt changes using the same projects and criteria.
