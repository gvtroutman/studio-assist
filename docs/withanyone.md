# Family photos with WithAnyone

Choose **Family photo (WithAnyone)** in Image Studio's Model menu. Select one to
four identities with a clear, individual face photograph in each profile and
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
pasting and the visual critic's automatic redraw do not run afterwards. Missing
photos, invalid positions and more than four people fail before rendering.

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
the likeness was weaker than the user wanted (2026-09-27: "not as strong as i'd
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
