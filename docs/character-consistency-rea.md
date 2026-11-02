# Character reference consistency

2026-10-14. Two reference paths could diverge between scenes or finishing stages:
Scene Builder cleared character item pictures along with the form's outfit,
and the final FaceFusion selection used live profile photos even when the
scene or character had chosen a different primary photo or saved photo set.

## Changes

- Scene generation saves matching character item pictures in `scene_wear`, paired
  with each visible person's ID and head region. Tags match that person's own
  look and description. Character Wearing photos participate as named tags;
  changing an outfit does not restore the old clothes. Shared scene descriptions
  do not activate clothing references on everyone.
- The existing Klein item pass uses those explicit targets. It skips missing or
  ambiguous faces and overlapping person detections. After the mask is feathered,
  the final composite is confined to the detected wearer's box and excludes
  other detected person boxes. These are conservative rectangles; they are not
  a guarantee of semantic ownership when a detector misses someone. Pose, model,
  face and clothing references remain separate.
- History keeps the item references and target settings. Fix clears them.
  Missing files, unavailable Klein/SAM3 support, and missing targets are reported;
  the existing cancellation and failed-pass recovery keep the preceding picture.
- Final FaceFusion sources follow the scene's primary photo and captured photo
  set, or a single form person's saved `face_photos`. Older scenes without those
  photos retain the profile fallback. Multiple form identities do not share an
  unassigned photo snapshot. The avatar remains display-only.

## REA evidence and limits

REA 6.4.0 inspected two ZIP snapshots containing the current Image Studio engine,
Scene Builder, item and head pass modules, and Z-Image/FLUX baseline workflow files.
Each inventory has six members. The complete evidence is kept locally in the
ignored `.work/rea-characters/` folder:

| Snapshot | REA evidence ID | ZIP SHA-256 |
| --- | --- | --- |
| Before | `ev_3471455392698f22b02b3bfcd3e6867d78114a8cd4891d9beaff5462edac5d97` | `5707da58689047ae47b9b06be77d923616b604da43c15b0b8e6c6f204539f73e` |
| Final | `ev_d4398e5f164b6e1503f4675f6fa5c1487faee12bd457706a754262ce5303d5ed` | `9729f322773b4edbd21dfb29be275c980e630703a8e235037b81155b5d3e8a71` |

These are artifact inventory observations, not Python semantic analysis or proof
of execution. The source-path findings and implementation choices above come
from repository inspection and the tests below. The snapshots also contain
other work already in progress in the shared workspace; their differences are
not all attributed to these consistency fixes. The installed REA process capture
adapter refuses native Windows capture because it cannot verify descendant
cleanup; the failed attempt is retained as `before-evidence.json`. No native
process capture or live image-generation observation is claimed.

The same offline probe, `.work/rea-characters/probe.py`, generated settings for two
characters in a workshop and then a garden. Before the change both scenes had no
item targets; afterward both had the red/blue shirt references paired with the
same corresponding person IDs and regions. This probe ran with ordinary Python,
not REA. The installed ComfyUI `comfy_extras/nodes_mask.py` was read to verify the
stock mask operations and their clamping behavior; no backend files were changed.

## Validation

`tests/test_character_consistency.py` covers per-person snapshots across scenes,
outfit overrides, tag scope, out-of-frame people, smaller explicit wearers,
missing/ambiguous targets, overlapping detections, final blend bounds after
feathering, full fake-client generation/history, missing files/backend support,
and final face-source ordering and legacy behavior.

The final integration suite passed 636 tests, including 17 new character tests:

```text
python tests/offscreen.py tests.test_scene tests.test_imagegen tests.test_character_consistency tests.test_wear tests.test_headswap tests.test_facefusion_profiles tests.test_image_pipeline tests.test_withanyone tests.test_identity_recipe tests.test_finish_line tests.test_surface -q
```

No live render comparison was performed. These checks establish source and
compositing behavior, not improved visual likeness, garment quality, or measured
consistency scores. The existing identity evaluation recipe still applies to
any claim of likeness. Reopen Studio Assist to load the updated Python modules.
