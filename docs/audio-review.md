# Audio Review

Open **File → Audio Review…** to compare alternative narration performances.
Choose an audio-capable inference host and its model ID. This is a separate
connection from Studio Assist's text/vision model; no model is downloaded or
loaded on the editing workstation.

The host must accept OpenAI-compatible **Chat Completions** requests with
`input_audio` WAV content and `modalities: ["text"]`. A text-only endpoint,
transcription server or ElevenLabs generation connection does not supply this
capability. Rejection of audio input is reported; no transcript-based fallback
or automatic retry is made.

1. Enter the host URL (including `/v1`, or the full `/chat/completions` URL)
   and audio model ID. An API key is optional and never saved.
2. Set a source start/end range, then choose **Add files…**. Blank end uses the
   full recording. Add the same file again with a different range to compare
   takes recorded together. Existing exported take WAVs can be added together.
3. Add two to six takes of the same line, up to 60 seconds each. WAV sources
   are sliced without changing their audio samples. MP3 requires FFmpeg on
   PATH, decoding the selected range to 48 kHz mono PCM without normalization.
4. Optionally enter the intended words and desired delivery. Click **Compare
   takes** to send the selected recordings to the chosen host. Host/model
   preferences are remembered; source files, script and API key are not.
5. Read the ranking, strengths, concerns and evidence timestamps. A tie remains
   a tie. Use **Save review…** for a text report or JSON with exact source ranges.

Recommendations are the audio model's subjective assessment. Source media and
Premiere/Resolve timelines are never changed. Evidence timestamps refer to
the selected excerpt; source offsets are retained separately. Invalid take IDs,
missing comparisons, inconsistent winners, or evidence outside a recording are
rejected rather than converted into selections.

Defaults may be supplied through `STUDIO_AUDIO_BASE_URL`, `STUDIO_AUDIO_MODEL`
and `STUDIO_AUDIO_API_KEY`. Keys stay in memory or the environment, not in
preferences, saved reports, error messages or transcripts. Redirects are refused
so recordings and authorization are not forwarded to another endpoint.

Reads/uploads are bounded (32 MB per source and combined audio, 128 KB reply,
120-second host timeout). Stop discards late results; it cannot retract an
already-sent host request. Closing the window cancels its timer and clears its
key field. FFmpeg children use the existing process containment and hidden window.
