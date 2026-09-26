# Verified ledger

What has been checked and found bug-free, so it isn't re-tested every session.
Re-check an entry only when a later change touches its code; then move it to
"Needs re-check". Put new testing effort into "Gaps".

## Verified

| Area | What was checked | Commit | Date |
| --- | --- | --- | --- |
| Head shape (`mq.HEAD_SHAPE`/`clean_head`/`head_warp`/`neck_scale`, `person_pieces(head=)`, hair follows it, `obj["head"]` saved, identity `head`, Head-panel sliders, Save to / Load identity, carried by `_set_character`) | `TestHeadShape` + `test_head_shape_sliders_save_to_the_identity_and_come_back`; scene + imagegen + pose 266 OK; renders checked by eye; live FLUX-dev A/B on the 5090 (same seed, no identity): silhouette carries at pose 0.85/depth 0.55 but the close-up is a mannequin; at 0.6/0.45 and 0.5/0.3 a real face and only neck/outline shift. NOT re-tested with the depth-relief change | branch head-profile | 2026-09-26 |
| Head depth in the face pass (`depth_values(window=)`, `depth_crop_png`, `head_box`, `face_graph` face `depth` → ControlNetApplyAdvanced at FACE_DEPTH_STRENGTH/END) | `test_a_face_crop_gets_its_own_head_depth`; scene + imagegen + critic 272 OK; live on the 5090 by script (not the app): one medium-shot base, same face redrawn at denoise 0.6 / depth 0.5 with neutral, broad, narrow head: broad visibly wider jaw and face, narrow narrower - but that base's head was turned otherwise than the mannequin's, and the change was mostly the redraw turning it (see the next row) | branch head-profile | 2026-09-26 |
| Face pass wired to head shape (`face_targets` facing/head/tall, `_head_depths`, `drawn_facing`/`facing_yaw`/`head_gate` via DWPose, HEAD_DENOISE, scene `head_depth` + slider) and shot-aware strengths (`framing`, FRAMING) | `test_a_head_shape_goes_into_its_face_redraw_as_far_as_the_turn_agrees`, `test_the_shot_sets_how_hard_pose_and_layout_hold`; 293 OK; live matrix through scene_maps → generation → job on the 5090 (Lilya, medium shot, one seed): framing at 0.75 gives real faces, no mannequin; the gate passed every face; but head depth 0 → 0.5 at denoise 0.6, and 0.8 at 0.75, left each face near-identical - a face already turned as the mannequin is is NOT reshaped by the face pass | branch head-profile | 2026-09-26 |
| Pick person (`people_found`, `pick_box`, `cutout_region`, `cutout_graph`, `find_people`, `cut_person`) | unit tests + live run on the 5090: 2 people found, one cut out clean | uncommitted | 2026-09-26 |
| Scene Builder Enrich (`suggest`, `read_suggestion`, `enrich_messages`, `enrich_answer`, words, save/load) | `EnrichTest` (incl. `place_suggestion`: every WHERE in frame, no floor overlap, no hiding/covering subjects, fallback, passer-by not a subject) + live runs on qwen3-coder-30b, frame rendered and checked | uncommitted | 2026-09-26 |
| Scene Builder shapes and props (`lathe`, `sphere`, `cone`, `capsule`, `wedge`, `fit`, compound `UNIT`, `STAND_INS`, Library buttons) | `TestShapesAndProps` + window test; every shape and prop rendered to PNG and checked by eye | uncommitted | 2026-09-26 |
| Visual Critic (`studio_critic`, `Studio._refine`, `_correct`, `_regenerate`, `face_graph` whole-picture crop) | `tests/test_critic.py` against fake ComfyUI and fake vision; the critic's call run live on qwen2.5-vl-7b with two reference photos (window fitted 8,192 -> 32,768, picture read correctly, cut-off JSON salvaged); the full loop from the UI NOT yet run live | uncommitted | 2026-09-26 |
| Merge of scene-from-picture into main (one asset library, `face_graph` with per-face words/PuLID and the Critic's `edit`/`mask`/`head` crops, `_middle_in_frame` for Enrich) | full suite 838 OK; all 13 saved scenes open with every object | uncommitted | 2026-09-26 |
| Critic memory (`remember`/`recall`, `critic_memory.json`, kept details in every later `compose` prompt and the next `_refine` canonical state) | `tests/test_critic.py` MemoryTest + `test_a_kept_scene_detail_goes_into_the_next_picture`; test_critic + test_imagegen 134 OK; NOT run live | uncommitted | 2026-09-26 |
| From a picture: floor and walls made automatically (`picture_answer` indoors/walls, `picture_room`, `_pictured` pressing Make) | `test_an_indoor_photo_gets_walls_round_the_camera_and_people`, `test_an_outdoor_photo_has_no_walls`, the window's picture test (two texture jobs queued and worn); `tests.test_scene` 133 OK; the new question NOT yet run on the live vision model | uncommitted | 2026-09-26 |
| Always-on LoRAs (`clean_lora` `always`, `compose` stack) + Z-Image Turbo default model | `test_always_on_loras_join_every_picture_from_a_model_they_suit`, `test_z_image_is_the_default_model`; full suite 840 OK; the Library editor checkbox not clicked live | uncommitted | 2026-09-26 |
| Scene Builder Look at (`aim_head`, `aim_heads`, `eye_point`, `look_at` saved, L tool ring: `HEAD_POSES`/`head_pose`, Camera, Point..., Esc; Look at camera / Stop looking, marker) | `TestLookAt` + `test_look_at_opens_a_ring_of_head_poses`; scene suite 131 OK; not clicked live in the app | uncommitted | 2026-09-26 |
| More detailed mannequin (ears, shoulder/elbow/wrist/knee joints, thumbs) and props (car bumpers/lights/hubs/glass cab, chair back posts/rails/stretchers, table apron, bench feet/stretcher, fence caps/pickets, lathed lamp post, parasol ribs, lobed tree and bush) | scene suite 131 OK; rendered `png()` checked by eye; not seen in the app's viewport | uncommitted | 2026-09-26 |
| Mannequin hands: four fingers + thumb, controls `wrist_*_bend`/`fingers_*_curl`/`fingers_*_spread`/`thumb_*_curl`, `gripped` (stein, pretzel, accordion), pretzel prop, accordion keys moved to their right, "Playing the accordion" pose | `test_a_hand_takes_what_it_holds_unless_posed_by_hand`, `test_fingers_curl_toward_the_palm_and_the_wrist_bends`; scene suite 135 OK; open, fist, stein, pretzel, accordion rendered with `png()` and checked by eye; sliders not dragged live in the app | uncommitted | 2026-09-26 |
| Hands into generation: `hand_joints` (one source for mesh and map), `rigs` gripped + `sk["hands"]`, `pose_figures` `hand_points`, `studio_pose.render_figures` given hand points, `hand_words` in `posture_words` | `test_the_pose_map_carries_the_hands_and_the_words_say_them`; scene + pose suites 151 OK; pose map for accordion and stein people rendered and checked by eye; NOT yet generated live on the 5090 | uncommitted | 2026-09-26 |

## Needs re-check

| Area | Why (what changed) |
| --- | --- |
| Enrich live runs | `SHAPES`/`ENRICH_SYSTEM` now list the new shapes and props; no live model run since |

## Gaps (write tests here next)

- `studio_images_ui.py` — no test file.
- `studio_scene_ui.py` — no test file (mannequin: body, clothes, hats/glasses, hair, shoe shapes).
- `studio_chat.py`, `studio_ui.py`, `studio_files.py`, `studio_com.py`, `studio_cep.py` — no test files.
