# Midjourney quality investigation

2026-10-14. The requested target was Midjourney, with the goal of bringing
Image Studio's generated picture quality closer to it. This investigation
used the installed REA 6.4.0 skill and its actual MCP contract, then normal
repository inspection and offline pipeline tests.

## Evidence boundary

Midjourney's public documentation describes separate style references,
personalization, a stylize control and a Raw mode. These describe product
controls; they do not disclose the model weights or a reproducible sampling
recipe. References:

- [Version](https://docs.midjourney.com/hc/en-us/articles/32199405667853-Version)
- [Stylize](https://docs.midjourney.com/hc/en-us/articles/32196176868109-Stylize)
- [Edit Model](https://updates.midjourney.com/edit-model-for-v8/)

The ordinary in-app browser could read the public home page and identify its
app bundle, `clientSideEntry-727srb3z.js`. Direct HTTP requests for the page
and that observed bundle returned 403. A bounded REA browser scenario used
an isolated, empty temporary Chrome profile, visited only the public home
page, waited 1.2 seconds and captured its DOM and network responses. It did
not log in, generate images, click an access challenge or use account data.

| Operation | REA evidence ID | Observation |
| --- | --- | --- |
| `capture_browser_scenario` | `ev_bf74cce139a2c1fc7bf0d52301050302e7ad2799c182d4e835f793177ad58a5c` | The captured page was an access challenge. No generation implementation was obtained. |
| `export_web_scripts` | `ev_e06eae16eb63c1ef53830e8afe8b1ec2f34e08e23c0c0635f72345aad4a21da8` | Exported scripts belonged to the access challenge; they were not treated as Midjourney generation code or analyzed for model behavior. |

The capture reports incomplete event coverage, unfinished responses and
unavailable content. It establishes the acquisition limit of this particular
attempt, not absence of application code or a complete search of Midjourney.
The request, canonical capture, export manifest and source hashes remain
in the ignored `.work/rea-midjourney/` folder. No downloaded vendor code is
shipped with Studio Assist. The
[REA artifact workflow](https://github.com/morluto/rea/blob/main/docs/javascript-artifact-reconstruction.md)
describes the distinction between static evidence and runtime behavior.

## Implemented quality fixes

Repository inspection found two inconsistencies that work against a coherent
chosen look:

1. Face redraws requested a real photographic face with pores and
   imperfections, even for an illustration. Other finishing crops requested
   photographic detail without the chosen style. Finishing now carries
   bounded style and selected camera-chemistry context into the relevant face,
   eye, beard, smile, hand, glasses and spot redraws. The prompt describes the
   part being redrawn,
   rather than repeating unrelated objects or framing from the whole scene.
   Explicit picture media in a brief also reach finishing; a painting
   mentioned as an object in a photographed scene is not treated as its medium.
2. Per-person face redraws rebuilt the style from its library record and
   could restore a LoRA trigger omitted from the base render because that
   LoRA was incompatible, missing or at zero strength. Finishing now uses
   the effective style composed for the base render.

These are our implementation choices and source findings, not recovered
Midjourney algorithms. No Midjourney weights, sampler, private API, synthetic
stylize equivalent or automatic prompt embellishment was added.

## Validation and remaining limits

The 14 focused tests in `tests/test_render_style.py` check composed conditioning
and fake-backend
generation/finishing graphs, including medium recognition, bounded crop
context, selected illustration/monochrome/film styles, LoRA trigger gating
and spot fixes. The existing unstyled photographic prompts retain their
measured wording.

The combined style, imagegen, finish-line and image-pipeline suite ran 326 tests:
280 passed, 46 were skipped, with no failures or errors. The full output is
retained in `.work/rea-midjourney/offline-tests.txt`. Whitespace checks also pass.

An offline comparison at seed 4242 also verified that the initial prompt and
generation graph remain identical for No style, Illustration, Black and white
and SX-70. The selected styles change face and hand conditioning; No style's
face and hand prompts remain identical. The comparison uses a baseline
reconstructed by reversing only this task's exact patch, preserving other
working-tree edits. There was no independently captured pre-edit hash; the
inverse replacements, reconstructed source hash and comparison results are
retained locally. This compares instructions and graphs, not output pixels.

Reproduce the focused suite with the repository's hidden-desktop runner:

```text
python tests/offscreen.py discover -s tests -p test_render_style.py -q
```

No live visual comparison was possible in this run: the dedicated workstation
ComfyUI endpoint was not running, and the other backend shares the inference
GPU. No backend was started or inference models unloaded for this task.
Offline checks establish conditioning consistency; they do not establish
improved aesthetics, measured likeness or Midjourney quality equivalence.

Reopen Studio Assist to load the changed Python modules. Select the desired
style normally; finishing now follows it.
