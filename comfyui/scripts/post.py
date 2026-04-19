"""Post-production workflows for ComfyUI: audio stem separation, STT,
and in-graph multi-video concatenation.

These replace host-side ffmpeg assembly for music-video pipelines and add
the missing pieces for lipsync (extract vocals + transcribe with timing)."""
from core import WorkflowGraph


def extract_stems(audio_filename, model_name="MelBandRoformer_fp16.safetensors",
                  filename_prefix="stems"):
    """Split mixed audio into 2 stems via Mel-Band RoFormer.

    Sampler outputs 2 AUDIO streams. Order is model-dependent — for the
    default vocal model, stem0 is vocals and stem1 is instrumental, but
    verify by ear. Saves both as FLAC via ComfyUI's SaveAudio."""
    g = WorkflowGraph()
    model = g.node("MelBandRoFormerModelLoader", model_name=model_name)
    audio = g.node("LoadAudio", audio=audio_filename, audioUI="")
    sampler = g.node("MelBandRoFormerSampler", model=model[0], audio=audio[0])
    g.node("SaveAudio", audio=sampler[0],
           filename_prefix=f"{filename_prefix}_stem0")
    g.node("SaveAudio", audio=sampler[1],
           filename_prefix=f"{filename_prefix}_stem1")
    return g.to_dict()


def transcribe(audio_filename, model_size="large-v3-turbo",
               language="auto", filename_prefix="transcript"):
    """Whisper STT with word- and segment-level timestamps via the
    'Apply Whisper' custom node. Save SRT's file persistence isn't
    retrievable via /view (wrong name path), so we wrap all three outputs
    in ShowText nodes — the caller pulls them out of history.outputs and
    writes them locally.

    Outputs (captured in history.outputs[<node>]['text'], in graph-node
    ID order):
      - plain transcript
      - segments SRT (sentence-ish cues)
      - words SRT (per-word cues — use for lipsync alignment)

    Run on extracted vocals, not the full mix — cleaner alignment."""
    g = WorkflowGraph()
    audio = g.node("LoadAudio", audio=audio_filename, audioUI="")
    whisper = g.node("Apply Whisper", audio=audio[0],
                     model=model_size, language=language, prompt="")
    text, segments, words = whisper[0], whisper[1], whisper[2]
    srt_seg = g.node("Save SRT", alignment=segments,
                     name=f"{filename_prefix}_segments")
    srt_word = g.node("Save SRT", alignment=words,
                      name=f"{filename_prefix}_words")
    # ShowText captures each STRING into history.outputs[node]['text'].
    # Order matters — caller reads them in this order:
    g.node("ShowText|pysssss", text=text)       # plain transcript
    g.node("ShowText|pysssss", text=srt_seg[0]) # segments SRT content/path
    g.node("ShowText|pysssss", text=srt_word[0])# words SRT content/path
    return g.to_dict()


def concat_videos(video_filenames, audio_filename=None, fps=24.0,
                  trim_durations=None, trim_starts=None,
                  filename_prefix="concat", format="mp4", codec="h264"):
    """Stitch N pre-rendered videos end-to-end inside ComfyUI.

    video_filenames: list of video paths (uploaded to ComfyUI input/).
    audio_filename:  optional audio overlay (the original song) —
      per-clip audio tracks are always discarded.
    trim_starts:     optional list[float] of seconds to SKIP from the
      start of each video. Used with transitions: a scene that has a
      transition before it plays from T/2 in (scene's first T/2 are
      replaced by the transition's last half).
    trim_durations:  optional list[float] of seconds to keep AFTER the
      trim_start offset. Drops lipsync buffer tails; with transitions,
      also drops the part of a scene that's replaced by the transition
      on the next boundary.

    Timeline math:
      final_len = sum(trim_durations). Each scene loses T/2 per
      transition boundary, each transition adds T, net zero — song
      timeline stays locked."""
    if not video_filenames:
        raise ValueError("video_filenames must be non-empty")
    g = WorkflowGraph()
    images_per_vid = []
    for i, vf in enumerate(video_filenames):
        vid = g.node("LoadVideo", file=vf)
        comp = g.node("GetVideoComponents", video=vid[0])
        images = comp[0]  # IMAGE batch; per-clip audio dropped
        start = 0
        if trim_starts and i < len(trim_starts) and trim_starts[i] > 0:
            start = int(round(float(trim_starts[i]) * float(fps)))
        if trim_durations and i < len(trim_durations) and trim_durations[i] > 0:
            num_frames = int(round(float(trim_durations[i]) * float(fps)))
            trimmed = g.node("GetImageRangeFromBatch",
                             images=images, start_index=start, num_frames=num_frames)
            images = trimmed[0]
        elif start > 0:
            # skip-only — drop frames before `start`, keep everything after
            trimmed = g.node("GetImageRangeFromBatch",
                             images=images, start_index=start, num_frames=100000)
            images = trimmed[0]
        images_per_vid.append(images)

    if len(images_per_vid) == 1:
        images_out = images_per_vid[0]
    else:
        batch_inputs = {"inputcount": len(images_per_vid)}
        for i, img in enumerate(images_per_vid):
            batch_inputs[f"image_{i+1}"] = img
        batch = g.node("ImageBatchMulti", **batch_inputs)
        images_out = batch[0]

    if audio_filename:
        aud = g.node("LoadAudio", audio=audio_filename, audioUI="")
        created = g.node("CreateVideo", images=images_out,
                         fps=float(fps), audio=aud[0])
    else:
        created = g.node("CreateVideo", images=images_out, fps=float(fps))

    g.node("SaveVideo", video=created[0], filename_prefix=filename_prefix,
           format=format, codec=codec)
    return g.to_dict()
