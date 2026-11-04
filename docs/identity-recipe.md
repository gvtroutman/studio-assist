# Identity recipe: establish likeness before changing the scene

This is the working evaluation procedure, not a claim that a training recipe has
already been validated. The initial evaluation did not establish likeness.

1. **Curate originals.** Inspect every photo for a clearly visible subject, focus,
   filters, occlusion and useful angle. Group near-duplicate frames from the same
   capture; count independent views, not files or crops. Record exclusions and
   why. Select a clear primary portrait. Keep original photos unchanged.
2. **Separate evaluation references before training.** Reserve independent captures
   where available; never put a crop or another frame from the same capture on the
   other side of the split. If none exist, label the trial as a training-reference
   comparison. Do not invent a minimum photo count or claim held-out validation.
3. **Freeze a trial.** Record selected source paths, captions, model, trigger,
   training settings and checkpoint names. Caption changing clothing, expression,
   background and lighting. Use the description as a supplement to visual
   conditioning, not as proof of identity. Do not silently increase training time
   after a failure. Change one variable per subsequent trial.
4. **Portrait first.** Use familiar hair and glasses, neutral light and a simple
   background. Compare no adapter against each checkpoint with identical settings
   at two or more fixed seeds. Disable pooling, face swap, style LoRAs, face redraw
   and refinement so they cannot obscure the checkpoint's contribution.
5. **Review likeness explicitly.** Compare face width/length, cheek volume, jaw,
   chin, eye spacing, nose, mouth and glasses against the originals. Record a
   reviewer, pass/fail and concrete discrepancies for every candidate/seed.
   Successful rendering is never a likeness pass. Reject attractive strangers.
6. **Change one thing at a time.** Only checkpoints passing all portrait seeds
   advance. Test an expression, then clothing, then background in separate prompts
   retaining the same simple framing. Review again before trying the full scene.
7. **Promote deliberately.** A reviewed checkpoint and its tested strength can be
   assigned to the profile. Archive failures and exact settings. Neither the
   evaluation command nor completion of training approves a profile. (The
   identity editor's **Build LoRA** button does set
   its finished LoRA as the person's Identity LoRA; that is a convenience, not a
   likeness pass. Reviewing it here still applies.)

## Executable evaluation

`tools/identity_recipe.py` runs steps 4–6 through the normal Image Studio engine
in an isolated library. It requires already installed FLUX adapters; it does not
train, install weights, or change the user's saved library. `plan` is offline.

```powershell
Copy-Item recipes/example-identity.json .work/identity-recipe.json
# Edit the local copy with your adapter filenames and prompts before running.
python tools/identity_recipe.py plan .work/identity-recipe.json portrait
python tools/identity_recipe.py run .work/identity-recipe.json portrait .work/portrait-recipe
```

The new output folder contains a frozen recipe, generation history and
`report.json`. Each result starts with `review.likeness: "pending"`. After actual
visual review, record `pass` or `fail`, `reviewer`, and explanatory `notes` there.
Do not auto-fill passing reviews. The report's `likeness_validated` remains false:
it is not an automatic certification or an independent benchmark score.

```powershell
python tools/identity_recipe.py plan .work/identity-recipe.json scene --portrait-report .work/portrait-recipe/report.json
python tools/identity_recipe.py run .work/identity-recipe.json scene .work/scene-recipe --portrait-report .work/portrait-recipe/report.json
```

The scene command checks that the portrait report belongs to the same recipe and
that each advancing checkpoint completed with an adapter and an explicit passing
review at every seed. It includes a no-adapter baseline again. A changed recipe
requires a fresh portrait comparison. These gates apply to this evaluation tool,
not the ordinary Image Studio Generate button. JSON review records are a local
workflow record, not tamper-proof attestations.

## Local recipes

Copy `recipes/example-identity.json` to `.work/identity-recipe.json`. Choose your
installed candidate adapter and write prompts from your own references. The example
is a template, not a trained identity or a validated likeness. Keep personal
captions, source ordering, photographs and training scripts outside Git.
