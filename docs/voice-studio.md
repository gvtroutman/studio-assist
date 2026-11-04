# Voice Studio

Open **Voice Studio** from Studio Assist's **+** tab menu. It shares a script,
voice reference and delivery brief across several free speech engines, saves
separate takes, and lets you listen, compare actual audio and export your choice.
The desktop remains stdlib-only. Speech models run on the LLM PC.

## Remote setup

The tab's **Remote setup...** button exports a ZIP with the remote service,
worker, setup scripts, this guide and a SHA-256 manifest. It contains no models,
credentials or third-party source. Extract it into a new folder on the LLM PC.
Do not run the engine installer on the editing workstation.

Install a separate **Python 3.11** on the LLM PC if it does not already have one.
VoxCPM's pinned TorchCodec dependency needs an FFmpeg **shared** build, version
4–7, whose DLLs are on that PC's `PATH`; see the [versioned installation guide](https://github.com/pytorch/torchcodec/blob/v0.7.0/README.md#installing-torchcodec).
From the extracted folder in PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\apps\voice_studio\setup_remote.ps1 -Python python -Engines voxcpm,qwen,omnivoice
.\apps\voice_studio\start_remote.ps1 -BindAddress YOUR_TAILNET_IP
```

`-Python` can be the absolute path to that Python executable. The first command
allows the bundled scripts for this PowerShell session only. Setup creates one
environment per engine group and installs pinned top-level packages.
It does not change ComfyUI or LM Studio's environments. Additional choices are
`chatterbox`, `kokoro`, and `pocket`; rerun setup with those engine names to add
them. `-CpuOnly` selects CPU PyTorch; Kokoro and Pocket always use CPU in the
automatic setup. Package resolution and live inference still need verification
on the remote machine; these environments were not installed on the workstation.
Setup runs dependency, import and CUDA checks before recording each environment.
These checks do not download model weights or prove speech generation works.

Replace `YOUR_TAILNET_IP` with the LLM PC's tailnet address. The start script
requires an explicit bind address and launches a hidden background server on
port 17494. Remote listeners require the generated token in
`voice-token.txt`. Enter that token in the tab, alongside the server URL, and click
**Connect**. The workstation remembers the URL; it does not save the token.
Startup errors go to `voice-server-error.log` in the extracted folder. The log
does not print the token. Keep this service private to the tailnet.
If Connect times out, check that log and that Windows Firewall allows TCP 17494
on the LLM PC's tailnet interface.

Models download on the remote PC only when **Allow remote model downloads** is
checked for a generation. Otherwise workers use offline Hugging Face mode and
report missing files. Pocket's current model download may require accepting its
Hugging Face access conditions and authenticating on the remote PC. There are no
paid API integrations. Each model and dependency retains its own license.

For a foreground server, useful for setup diagnostics:

```powershell
$env:STUDIO_VOICE_TOKEN = (Get-Content .\voice-token.txt -Raw).Trim()
python -m apps.voice_studio.remote --host YOUR_TAILNET_IP --port 17494
```

Stop only the server PID reported by the start script, or press Ctrl+C when
running in the foreground. The service contains every worker in a process job
and ends that owned worker tree on shutdown or cancellation.

## Engines and voice control

| Choice | Role | Reference | Delivery control |
| --- | --- | --- | --- |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) | Voice design and controlled cloning | Optional; add its transcript for continuation cloning | Written description, including with a reference |
| [Qwen clone](https://github.com/QwenLM/Qwen3-TTS) | A voice from a recording | Required, with transcript | Follows the recording; no free-text style instruction |
| [Qwen presets](https://github.com/QwenLM/Qwen3-TTS) | Expressive preset speakers | Unused | Written instruction; speaker menu selects the preset |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) | Multilingual cloning or voice design | Optional; transcript required when cloning | Description for voice design; cloned prompt follows the recording |
| [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) | English cloning and vocal tags | Required, longer than five seconds | Inline tags such as `[laugh]`; no free-text style prompt |
| [CosyVoice 3](https://github.com/QwenAudio/CosyVoice) | Multilingual reference speech | Required, with transcript | Its instruct mode accepts the delivery brief |
| [Kokoro](https://github.com/hexgrad/kokoro) | Small preset narrator | Unused | Preset selected for the language |
| [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) | CPU reference speech | Optional | Reference or default voice; English is exposed here |

Qwen, VoxCPM2, OmniVoice, CosyVoice and Kokoro publish Apache-licensed code/model
releases; Chatterbox publishes MIT releases. Pocket's code is MIT and its current
[model card](https://huggingface.co/kyutai/pocket-tts) lists CC BY 4.0. The adapter
catalog records the upstream sources and package pins checked on 2026-10-15.
The code integrates their public Python APIs; it does not merge model weights.

CosyVoice needs its own source checkout and environment following the official
installation instructions. It is not included in automatic pip setup. Add an
entry to the remote `voice-environments.json`, preserving existing entries:

```json
{
  "cosyvoice": {
    "python": "D:/VoiceModels/cosyvoice-env/python.exe",
    "source": "D:/VoiceModels/CosyVoice",
    "device": "cuda:0",
    "models": {"cosyvoice": "D:/VoiceModels/Fun-CosyVoice3-0.5B"}
  }
}
```

Those are example paths. Use the actual environment, source checkout and model
directory on the remote PC. The tab lists an engine as **configured** when its
Python executable exists; this does not assert that model weights are cached or
that GPU inference has been tested. An unconfigured engine cannot be generated.

## Generate and choose a performance

1. Enter a script and delivery brief. For a cloned voice, choose a clean 6–10
   second **16-bit PCM WAV** and enter the words spoken in it. Save a named voice
   profile to reuse these fields. A preset engine uses its own voice.
2. Select complementary configured engines and one or two takes per engine,
   with at most six takes in a batch. Different engines and presets may produce
   different voices; use reference engines together when comparing one identity.
3. Click **Generate takes**. Jobs run serially. Each worker loads one engine,
   generates the passages and exits, releasing its model memory. Idle LM Studio
   models on the same host are unloaded first, using the Image Studio's lazy
   reload policy. Models serving another busy chat are retained. This is not a
   reservation across independent ComfyUI jobs; avoid competing GPU workloads.
4. Select a take and click **Play**, or double-click it. Peak measurements can
   reveal silence or clipping; they do not rate naturalness or likeness. Long
   scripts split near sentence boundaries, with an 80 ms silence between chunks.
   Voice continuity and chunk boundaries still need a listening check.
5. Select matching-script takes and click **Compare selected audio** to prefill
   Audio Review. It requires a separately configured audio-capable inference
   host. No comparison starts automatically. For takes longer than 60 seconds,
   the first 60 seconds are preselected. The ranking is subjective and applies
   only to the attached excerpts. Manual audition and export work without it.
6. **Export selected...** copies the chosen WAV exactly. Original takes remain
   separate and are not automatically denoised, normalized, crossfaded or mixed.

## Recovery and limits

Drafts, named profiles, pending remote job IDs and take metadata live under
`voice-studio/` beside Studio Assist's settings. Local WAVs include a checksum and
their script, engine, seed and reference-profile name in the library metadata.
Up to 200 recent takes appear in the list; older audio is retained on disk.

If a connection drops after submission, use **Recover jobs** with the same URL
and token. Recovery only inspects existing IDs and downloads completed output;
it never resubmits a generation. Pending entries include planned jobs that may
never have reached the server. A missing ID is reported and remains available
for inspection in `library.json`; it is not silently turned into a new job.

The remote service persists statuses and WAVs in `voice-jobs/<job-id>/`. Completed
jobs survive a server restart. An interrupted job becomes failed and is not
replayed. Per-job `worker.log` and `request.json` support diagnosis. It retains
at most 256 jobs per data folder; archive the folder and select a new one with
`--data-dir` when full. No background cleanup deletes reference or output audio.

Requests cap scripts at 12,000 characters, references at 8 MB / 30 seconds, output
at 64 MB / 20 minutes and inference at 30 minutes. Remote errors or an uncertain
submission stop the batch. Closing the tab saves its draft and requests worker
cancellation; failed saves veto closing. Workers cannot update a closed tab.

Offline protocol, cancellation, persistence and GUI checks use synthetic WAVs.
They do not establish speech quality, identity similarity, package compatibility
or GPU performance on the LLM PC. The remote endpoints did not answer during
initial setup checks on 2026-10-15; live generation remains unverified.

The 22 Voice Studio tests and 16 Audio Review tests passed. The 307 application
tests passed in a separate run. A combined 345-test run still reports one Tk
transcript-selection error; that test passes alone. This suite interaction is
unresolved and is separate from the remote model checks above.
