# Family photos with WithAnyone

Choose **Family photo (WithAnyone)** in Image Studio's Model menu. Select one to
four identities with clear, individual face photographs in each profile and
describe the activity. Without a scene, people are placed left to right in the
selection order. A reference containing several faces is refused; crop it to
the intended person first.

For an existing Scene Builder scene, give each visible person a **Face** photo
(or a character with an identity reference), choose the same model, and generate.
The recipe uses each person's projected head region to keep the photo and
position paired. Scene descriptions supply clothes, expressions and activities.
This version uses face positions and text, not pose/depth ControlNets or a source
image. Body poses and prop placement therefore remain approximate.

The output is the WithAnyone generation itself. PuLID, face detail, real-face
pasting, FaceFusion, finishing passes and the visual critic's automatic redraw do
not run afterwards, even when a selected profile has Final face swap enabled.
FaceFusion is not required for this recipe. The first saved reference photo is
used per person. Scene Builder also reads the linked identity's library, with
its explicitly selected face photo used first. Other photos remain in the library.
Avatars are
display-only and never included. Missing
photos, invalid positions and more than four people fail before rendering.

## Combining photographs of one person

**Disabled by default after failed likeness review (2026-09-27).** Gavin rejected
the five-photo result as not looking like Lilya. Passing execution tests and
producing one person did not validate likeness. The developer-only
`experimental_reference_groups` setting retains the experiment for controlled
comparisons; it is not an approved identity-preservation method. The normal form
uses the single-reference path and says so. Do not re-enable blending or describe
it as a likeness fix based on structural tests alone.

`StudioWithAnyoneReferences` links each person's extra photos to their own
sampler input. Photos keep their original dimensions until face extraction.
Each photo must contain exactly one detected face; an invalid photo fails with
the person and photo number rather than silently disappearing from the group.
History records every source photo and the grouping in the workflow.

The node extracts ArcFace and SigLIP features for each photo separately and gives
all of that person's reference tokens the same target face region in upstream's
attention mask. Extra photos do not create extra person positions. This is a
Studio Assist extension to the upstream one-photo-per-person interface, not a
trained multi-view identity model or a promise that more photos always improve
likeness. It does not average photographs or embeddings. Consistent, clear photos
of the same person are needed; mismatched views or appearance can conflict.

Updating from the single-photo node requires copying both `__init__.py` and
`references.py` to the backend's `custom_nodes/studio_withanyone`, then restarting
ComfyUI when idle. An old backend is refused for grouped jobs, never silently
reduced to one photo. Restart Studio Assist to load the new planner.

`tools/compare_withanyone_references.py --identity <id> --output <new-folder>
--prompt <text>` runs a first-photo/all-photos comparison with the same seed,
prompt and sampler settings, in an isolated library. It leaves the user's library
and previous pictures unchanged and refuses a busy ComfyUI queue.

## Primary appearance with pooled identity (experimental)

Enable **Pool photos for WithAnyone (experimental)** in People → Image references
for each identity that should use its whole set (up to eight photos). Leave it
off to use that person's primary only. The switch also follows a linked identity
into Scene Builder. Different people's feature sets stay separate. The planner
uses `StudioWithAnyonePooled` so an older backend is explicitly refused; install
the updated node and restart ComfyUI before using it. The detailed description
continues to guide the prompt with either setting. This is an opt-in experiment,
not a claim of improved likeness. The previous separate-token experiment is not
what this switch enables.

The node also accepts `reference_mode="identity_consensus"`. Each photo still
must contain exactly one detected face. The primary supplies the sole SigLIP
appearance feature map; the normalized ArcFace directions from all photos are
averaged, normalized again, and scaled to the mean input norm. There is one
feature set and one target region per person, irrespective of photo count.
This avoids placing conflicting patch grids from different expressions and
views into the same attention region. It is an untrained pooling experiment,
not an established likeness improvement; normal generation remains unchanged.

The pinned vendor model concatenates ArcFace features into its SigLIP branch
(`lq_in_sig`), so `siglip_weight=1` still uses the identity vector. The separate
ArcFace attention branch has zero weight at that setting, not all ArcFace input.

`tools/validate_identity_guidance.py` uses the installed ComfyUI Python and
vendor files with the repository's node, without installing or restarting it.
It renders primary-only, separate-photo tokens, and pooled identity under the
same prompt, seed, primary photo, region, resolution and sampling settings.
It refuses a busy server and writes its results to a fresh folder. This direct
node trial does not validate deployment or the GUI job path. Likeness requires
visual review of the outputs alongside the original photos.

## Staged identity recipe

Use [the identity recipe](identity-recipe.md) to establish portrait likeness before
testing scene changes. Its reusable evaluator compares existing LoRA checkpoints
with a baseline at fixed seeds and requires recorded likeness reviews before
advancing. Pooling remains experimental; training or render completion is not a
likeness pass.

## Detailed identity descriptions

Identity profiles now have an editable `description`, separate from private
notes and the scene prompt. Describe stable visible traits in concrete prose:
face proportions, forehead and jaw, cheeks and chin, eye and brow shapes, nose,
mouth, hair and distinctive accessories. Keep uncertainty explicit and avoid
inferring ancestry, health, personality or other traits not visible in photos.
Clothing, pose and expression belong to the requested scene. Descriptions are
included for selected profiles, or for the linked Scene Builder identities
when a scene is supplied, with person/position labels. History retains the
final prompt and form profiles' descriptions.

The validation runner's `--description <text-file>` uses the same prompt helper
to test description-on against description-off, without editing the profile.
Long prose is not a replacement for photographic identity conditioning.

### Controlled trial, 2026-09-27

Five local renders used Lilya's five saved references, photo 4 as primary,
seed 240927, 768 square, 25 steps, guidance 4 and SigLIP weight 1. The first
three compared primary-only, separate tokens and pooled identity. Two more
added the detailed description to primary-only and pooled identity. All five
completed; 22 WithAnyone tests and 167 Image Studio tests passed.

Visual inspection found only modest changes from pooling and description.
The generated face remained narrower and the glasses larger/heavier than the
primary reference. This is not a validated likeness fix. The description field
is available in the profile editor; pooling remains a node-level experiment,
not the form's default. The draft description was tested without modifying the
saved profile. Results: `.work/identity-guidance-consensus-v2`,
`.work/identity-guidance-described`; comparison and exact description:
`.work/identity-guidance-review.html`. No backend installation was changed.

## Backend installation

Studio Assist remains stdlib-only. All neural-network code runs inside the
ComfyUI backend. Run its Python against `tools/install_withanyone.py` with
`--comfy`, `--models`, and optionally `--library` pointing at the Image Studio
library. The installer adds this model to an existing library without changing
other models or the selected recipe. Restart ComfyUI when idle, then reopen
Studio Assist to load the new code and model entry.

The installer reuses FLUX.1-dev, CLIP-L, T5, the FLUX VAE and antelopev2. It
downloads WithAnyone and SigLIP and copies a pinned upstream implementation into
the custom node's `vendor` folder. It does not install or upgrade Python packages.
Existing dependencies must include torch, transformers, insightface,
onnxruntime, OpenCV, einops, safetensors and huggingface_hub.

The initial model record is enabled on the 32 GB 5090; the 24 GB 3090 is marked
unavailable until a lower-memory implementation is tested. Text encoding,
reference detection and SigLIP run on CPU. The diffusion model is loaded for one
render and moved off the GPU in `finally`; progress checks cancellation every
sampling step. The VAE decodes in tiles. There is no resident model cache in this
node, so repeat renders pay the model load cost.

## Source and recipe

- [Original WithAnyone project](https://github.com/Doby-Xu/WithAnyone)
- [Model weights](https://huggingface.co/WithAnyone/WithAnyone)
- [ComfyUI port](https://github.com/okdalto/ComfyUI-WithAnyone), pinned to
  `6bb610caabc65e7b2fd37eb7eeb6cc5392347f36`.

`comfy_nodes/studio_withanyone` wraps the port's FLUX pipeline with explicit
photo selection, local model paths, CPU reference encoders, cancellation and
GPU cleanup. The upstream license is copied alongside its code. The recipe uses
25 steps, guidance 4, and SigLIP weight 1.0 (ArcFace `1 - siglip_weight`, so 0).
All settings and uploaded-reference paths are saved in normal Image Studio History.

The SigLIP weight is upstream's "Resemblance in Spirit <-> Resemblance in Form"
slider. SigLIP carries the reference's face as it looks (shape, expression,
makeup, glasses, hair); ArcFace carries only an identity vector and loses hair,
skin, age and build unless the words say them. Upstream's demo defaults it to
1.0 and says identity is also better kept that way. At the ComfyUI port's 0.8
the likeness was weaker than Gavin wanted (2026-09-27: "not as strong as i'd
like"), so the recipe now uses 1.0. Not yet measured on the 5090. Lower it in the workflow's `defaults` for
more freedom (a stylised picture, a changed hairstyle). The node's own default
changed with it; an older node copy on a backend still takes the workflow's
value, so reinstalling is not needed for the change.

`tools/try_withanyone.py` makes an isolated trial through the same job executor:
pass one or more `--ref` paths, `--prompt`, and `--output`. It refuses to run
over an existing ComfyUI queue and stores results in that output folder.

## Local validation, 2026-09-26

A 768 x 768, 25-step Oktoberfest portrait completed on the 5090 in 61.9 seconds
of backend execution. GPU free memory returned to approximately 32.3 GB after
cleanup. A separate 512 x 512, two-step run exercised two reference inputs using
the same photo in both slots; this verifies the multi-person tensor path, not
likeness separation between different family members. That still needs a trial
with their individual photographs.

## Grouped-reference validation, 2026-09-27

Execution-only validation; the grouped result subsequently failed the user's
likeness review. The 5090 completed a same-prompt, same-seed comparison at 768 x 768 and 25 steps:
the first saved Lilya photo versus all five saved photos. The grouped render took
74.6 seconds end to end, produced one person, recorded all five sources, and ran
no FaceFusion or finishing redraw. Both pictures were visually inspected. This
validates the live multi-photo path, not likeness consistency across many seeds
or poses. Results and histories are in `.work/withanyone-library-comparison`.
GPU free memory returned to about 30.1 GiB after cleanup. The 17 WithAnyone tests
and 164 Image Studio tests passed, including six photos bound to one region,
separate groups for separate people, old-backend refusal, and model-specific UI
labelling.

Follow-up after rejection: the first-photo and grouped outputs' detected face
boxes were both fully inside their conditioning region (coverage 1.0), so region
clipping did not explain this failure. A controlled single-reference render using
the clearer frontal photo (`5939d9550db20a49.png`), the same prompt and seed,
also changed facial proportions. It is stored in `.work/withanyone-frontal-control`
and is not an approved likeness. No successful likeness fix has been established.
Do not describe another completed render as resolving this issue without the
user's likeness judgment.
