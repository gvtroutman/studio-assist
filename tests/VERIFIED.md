# Verified ledger

What has been checked and found bug-free, so it isn't re-tested every session.
Re-check an entry only when a later change touches its code; then move it to
"Needs re-check". Put new testing effort into "Gaps".

## Verified

| Area | What was checked | Commit | Date |
| --- | --- | --- | --- |
| Pick person (`people_found`, `pick_box`, `cutout_region`, `cutout_graph`, `find_people`, `cut_person`) | unit tests + live run on the 5090: 2 people found, one cut out clean | uncommitted | 2026-09-26 |
| Scene Builder Enrich (`suggest`, `read_suggestion`, `enrich_messages`, `enrich_answer`, words, save/load) | `EnrichTest` (incl. `place_suggestion`: every WHERE in frame, no floor overlap, no hiding/covering subjects, fallback, passer-by not a subject) + live runs on qwen3-coder-30b, frame rendered and checked | uncommitted | 2026-09-26 |

## Needs re-check

| Area | Why (what changed) |
| --- | --- |

## Gaps (write tests here next)

- `studio_images_ui.py` — no test file.
- `studio_scene_ui.py` — no test file (mannequin: body, clothes, hats/glasses, hair, shoe shapes).
- `studio_chat.py`, `studio_ui.py`, `studio_files.py`, `studio_com.py`, `studio_cep.py` — no test files.
