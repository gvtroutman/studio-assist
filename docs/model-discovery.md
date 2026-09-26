# Model and code discovery

In Image Studio, open **Hugging Face** or **Civitai** beside Manage.
Opening either window fetches candidate metadata on a background worker. A
successful result is reused for 24 hours; **Refresh now** bypasses that cache.
This checks when the window opens, not on a timer while Studio Assist is closed.

- Hugging Face searches FLUX, Z-Image and Qwen-Image, ordered by downloads.
- Civitai checks popular LoRAs and checkpoints, retaining families for which
  Studio Assist has workflows. Exact model, license and hardware compatibility
  still need review; popularity is not proof of improved output.
- Both windows check the latest published releases of ComfyUI, Diffusers and
  FaceFusion. These are upstream candidates, not comparisons against installed
  versions or a guarantee that the release fixes a Studio Assist issue.

**Review source** opens the provider page. **Keep link** saves a model link.
**Import LoRA** opens the existing Civitai importer with that candidate filled in;
choose the destination there and press Import. Hugging Face downloads and
checkpoint installation still follow the provider's instructions.
Three bundled, supported add-ons appear first, even when discovery is offline:
color matching for repaired areas, pose extraction from photos, and reference-face
blending. Their descriptions explain what each does, its benefit and requirements.

**Add to app** asks for the ComfyUI folder containing `main.py` and
`folder_paths.py`. It installs that bundled node into `custom_nodes`, verifies the
copied bytes, and preserves any previous `__init__.py` in a uniquely named `.bak`
file beside it. Reinstalling identical code changes nothing. Linked destinations
are refused. Other files in the add-on folder are retained.

Installation copies node code only. Required packages and models listed on the
card must already be available in ComfyUI. Restart that backend when its queue is
idle, then use **Check connections**; ComfyUI must load the node before it can be
used. Selecting a folder on this PC does not install on a remote backend.

Upstream releases without a supported installer remain **Review only**. No coding
agent is invoked, no fetched code is executed and no packages are added to Studio
Assist's Python. The desktop app remains stdlib-only.

The provider's key is sent only to its own HTTPS origin. Cross-origin redirects
are refused; GitHub release requests receive no provider key. Keys are not written
to the discovery cache. Each response is capped at 2 MiB with a 15-second socket
timeout, using at most four concurrent requests. If feeds fail, available results
remain usable and old cached candidates are marked as previous results.

Implementation: `studio_discovery.py`, `studio_addons.py`; windows: `studio_images_ui.py`.
Tests: `tests/test_discovery.py`, `tests/test_addons.py` and `TestImageStudioTab` in `tests/test_imagegen.py`.

API references:
- https://huggingface.co/docs/hub/api
- https://developer.civitai.com/site/reference
- https://docs.github.com/en/rest/releases/releases#get-the-latest-release
