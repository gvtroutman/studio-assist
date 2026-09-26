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
| Visual Critic (`studio_critic`, `Studio._refine`, `_correct`, `_regenerate`, `face_graph` whole-picture crop) | `tests/test_critic.py` against fake ComfyUI and fake vision; NOT yet run live | uncommitted | 2026-09-26 |
| Merge of scene-from-picture into main (one asset library, `face_graph` with per-face words/PuLID and the Critic's `edit`/`mask`/`head` crops, `_middle_in_frame` for Enrich) | full suite 838 OK; all 13 saved scenes open with every object | uncommitted | 2026-09-26 |
| Always-on LoRAs (`clean_lora` `always`, `compose` stack) + Z-Image Turbo default model | `test_always_on_loras_join_every_picture_from_a_model_they_suit`, `test_z_image_is_the_default_model`; full suite 840 OK; the Library editor checkbox not clicked live | uncommitted | 2026-09-26 |

## Needs re-check

| Area | Why (what changed) |
| --- | --- |
| Enrich live runs | `SHAPES`/`ENRICH_SYSTEM` now list the new shapes and props; no live model run since |

## Gaps (write tests here next)

- `studio_images_ui.py` — no test file.
- `studio_scene_ui.py` — no test file (mannequin: body, clothes, hats/glasses, hair, shoe shapes).
- `studio_chat.py`, `studio_ui.py`, `studio_files.py`, `studio_com.py`, `studio_cep.py` — no test files.
