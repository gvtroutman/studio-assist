# Verified ledger

What has been checked and found bug-free, so it isn't re-tested every session.
Re-check an entry only when a later change touches its code; then move it to
"Needs re-check". Put new testing effort into "Gaps".

## Verified

| Area | What was checked | Commit | Date |
| --- | --- | --- | --- |
| Pick person (`people_found`, `pick_box`, `cutout_region`, `cutout_graph`, `find_people`, `cut_person`) | unit tests + live run on the 5090: 2 people found, one cut out clean | uncommitted | 2026-09-26 |
| Scene Builder Enrich (`suggest`, `read_suggestion`, `enrich_messages`, `enrich_answer`, words, save/load) | `EnrichTest` (incl. `place_suggestion`: every WHERE in frame, no floor overlap, no hiding/covering subjects, fallback, passer-by not a subject) + live runs on qwen3-coder-30b, frame rendered and checked | uncommitted | 2026-09-26 |
| Scene Builder shapes and props (`lathe`, `sphere`, `cone`, `capsule`, `wedge`, `fit`, compound `UNIT`, `STAND_INS`, Library buttons) | `TestShapesAndProps` + window test; every shape and prop rendered to PNG and checked by eye | uncommitted | 2026-09-26 |
| Visual Critic (`studio_critic`, `Studio._refine`, `_correct`, `_regenerate`, `face_graph` whole-picture crop) | `tests/test_critic.py` against fake ComfyUI and fake vision; the critic's call run live on qwen2.5-vl-7b with two reference photos (window fitted 8,192 -> 32,768, picture read correctly, cut-off JSON salvaged); the full loop from the UI NOT yet run live | uncommitted | 2026-09-26 |
| Merge of scene-from-picture into main (one asset library, `face_graph` with per-face words/PuLID and the Critic's `edit`/`mask`/`head` crops, `_middle_in_frame` for Enrich) | full suite 838 OK; all 13 saved scenes open with every object | uncommitted | 2026-09-26 |
| Always-on LoRAs (`clean_lora` `always`, `compose` stack) + Z-Image Turbo default model | `test_always_on_loras_join_every_picture_from_a_model_they_suit`, `test_z_image_is_the_default_model`; full suite 840 OK; the Library editor checkbox not clicked live | uncommitted | 2026-09-26 |
| Scene Builder Look at (`aim_head`, `aim_heads`, `eye_point`, `look_at` saved, L tool ring: `HEAD_POSES`/`head_pose`, Camera, Point..., Esc; Look at camera / Stop looking, marker) | `TestLookAt` + `test_look_at_opens_a_ring_of_head_poses`; scene suite 131 OK; not clicked live in the app | uncommitted | 2026-09-26 |
| More detailed mannequin (ears, shoulder/elbow/wrist/knee joints, thumbs) and props (car bumpers/lights/hubs/glass cab, chair back posts/rails/stretchers, table apron, bench feet/stretcher, fence caps/pickets, lathed lamp post, parasol ribs, lobed tree and bush) | scene suite 131 OK; rendered `png()` checked by eye; not seen in the app's viewport | uncommitted | 2026-09-26 |

## Needs re-check

| Area | Why (what changed) |
| --- | --- |
| Enrich live runs | `SHAPES`/`ENRICH_SYSTEM` now list the new shapes and props; no live model run since |

## Gaps (write tests here next)

- `studio_images_ui.py` — no test file.
- `studio_scene_ui.py` — no test file (mannequin: body, clothes, hats/glasses, hair, shoe shapes).
- `studio_chat.py`, `studio_ui.py`, `studio_files.py`, `studio_com.py`, `studio_cep.py` — no test files.
