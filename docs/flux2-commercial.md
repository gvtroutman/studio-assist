# Commercial FLUX.2 workflow

Choose **FLUX.2 Klein 4B distilled** in Image Studio's model menu after reopening
Studio Assist. Existing libraries receive the model once; deleting it stays a
deletion. The baseline uses the official BFL FP8 model, Qwen3 4B encoder and
FLUX.2 VAE, with Euler, four steps, CFG 1, `Flux2Scheduler` and
`EmptyFlux2LatentImage`. The negative is `ConditioningZeroOut` of the positive.
There are no added face, refinement or reference passes.

For ComfyUI directly, drag `workflows/flux2_klein4b_commercial.json` onto its
canvas. Change the Prompt, Width, Height and seed nodes, then run it. The
separate `flux2_klein4b_commercial_api.json` is for API clients. The Image Studio
adapter is `comfy_workflows/klein4b_distilled.json`.

## Files and licensing

| ComfyUI folder | File | Source |
| --- | --- | --- |
| `diffusion_models` | `flux-2-klein-4b-fp8.safetensors` | [BFL official FP8 release](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) |
| `text_encoders` | `qwen_3_4b.safetensors` | [Comfy's packaged encoder](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders) |
| `vae` | `flux2-vae.safetensors` | [Official ComfyUI recipe](https://docs.comfy.org/tutorials/flux/flux-2-klein) |

[BFL licenses Klein 4B, including the FP8 release, under Apache 2.0 and explicitly
allows commercial use](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
The 4B family is distinct from 9B and FLUX.2 dev in the library, license sorting
and filename recognition. The official baseline has no third-party LoRAs or
custom nodes. Klein 9B's non-commercial license remains correctly labeled.

The workstation model was installed into `D:/ComfyUI-models/diffusion_models`.
The encoder and VAE were already present. The model is pinned to revision
`5b4408e59397a4a37ccb46afe426d8ed86379441`, 4,070,624,520 bytes, SHA-256
`97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6`.
The 3090 library override initially marks this model absent there; install its
files and set that override in Models to use that backend.

## REA evidence and reproduction boundary

REA 6.4.0 analyzed the installed ComfyUI frontend 1.52.7's unchanged
`settingStore-CwNB8aKw.js` and its source map. Evidence:
`ev_55e5830c032ffb29ce65e2f319c98389b712e3c677f2aa611dcd886556bb5e89`.
There were 439,452 visited AST nodes, 2,901 findings and no parse failures.
Application inventory coverage is complete for this two-file subset; semantic
coverage is partial at the 100,000-node ceiling. This is static evidence.

REA authenticated the source-map originals `src/scripts/app.ts` and
`src/utils/executionUtil.ts`. Their SHA-256 values are respectively
`8de496f05a3e91cfff0c83be8d534122aec7e21a97170d09c5965c7540b9d4bd` and
`9ba4e7ceadda90f54389cc6c81331cbc590c79a24073356ef8f87643a13a134a`.
The latter describes expansion of executable subgraphs, omission of muted
nodes, serialization of widget values and resolution of links into API inputs.
Source contents were independently checked against REA's recorded digests.

The graph itself comes from the installed official template package
`comfyui_workflow_templates_json 0.1.95`,
`image_flux2_klein_text_to_image.json`, SHA-256
`f5b2e75448e1ef44ab3d08da00900000f0258f8f963934370c6a1c329d1328c2`.
The unchanged reference is retained in `tests/fixtures/flux2_klein4b_official.json`.
Its distilled subgraph is expanded into 16 native nodes. Internal node IDs gain
100 to avoid collision with the outer prompt node. The model filename changes
to BFL's official FP8 release; the seed is fixed at 0 for the example, and the
output prefix is `ImageStudio/flux2_klein4b`. All sampling and conditioning
connections are retained. REA does not infer model licensing or reconstruct
neural-network weights; licenses come from the official model cards.

Local analysis, source authentication, install and execution receipts are in
the ignored `.work/rea-flux2/` directory. No frontend source or large REA graph
is shipped in the repository.

## Validation

`tests/test_flux2_commercial.py` compares the reproduced edges to the official
reference, checks UI/API correspondence, template filling, shared dimensions,
family/license selection, missing-file refusal and durable library migration.
The existing template, library, pipeline and add-on suites also run: 341 tests
were checked, with 295 passing and 46 display-dependent tests skipped.

Live ComfyUI 0.37.2 validated all 16 nodes and produced the 1024-square hedgehog
example in 3–7 seconds on the 5090 (excluding server startup), using an isolated hidden server with
all custom nodes disabled. The saved PNG embeds the importable UI workflow.
The test server closes after validation. This proves one successful API run;
it does not establish identical pixels across precision formats or backend
versions. The expanded UI's links are checked offline; interactive frontend
import was not exercised.
