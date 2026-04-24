"""Post-production workflows for ComfyUI: audio stem separation, STT,
and in-graph multi-video concatenation.

These replace host-side ffmpeg assembly for music-video pipelines and add
the missing pieces for lipsync (extract vocals + transcribe with timing)."""
from pathlib import Path
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


COMFY_INPUT_DIR = "/app/ComfyUI/input"
COMFY_OUTPUT_DIR = "/app/ComfyUI/output"


def concat_videos_ffmpeg(staged_subfolder, audio_filename=None,
                          filename_prefix="concat_ffmpeg"):
    """Stream-copy N-clip concat using the ComfyUI-FFmpeg `MergingVideoByPlenty`
    node. Near-zero memory — ffmpeg concat demuxer with `-c copy`, no
    decode/re-encode — and it handles 20+ clips without OOM'ing the comfy
    server (unlike the BatchImagesNode path in `concat_videos`, which
    buffers every frame in RAM).

    `MergingVideoByPlenty` reads a FOLDER of videos and concats them in
    alphabetical order. The caller is expected to have uploaded all
    source clips into a single subfolder under `/app/ComfyUI/input`, with
    numeric filename prefixes (e.g. `001_*.mp4`, `002_*.mp4`) so the
    alphabetical order matches the intended sequence. See `stage_clips_for_concat`.

    Requires all input clips to have matching codec / resolution / fps.
    LTX-V's default h264 mp4 output at a fixed resolution satisfies this.

    Does NOT support per-clip trim — pre-trim clips at the caller if you
    need transition-style overlap cuts. For the trim+audio-overlay case
    fall back to the heavy `concat_videos`.

    Args:
        staged_subfolder: subfolder name under `/app/ComfyUI/input/` that
            already contains the clips (alphabetically-ordered filenames).
        audio_filename: optional server-side basename (in `/app/ComfyUI/input`)
            of an mp3 to overlay — stream-copy, replaces whatever audio
            tracks the clips carried. If None, clip audio is kept.
        filename_prefix: output filename stem. Final file lands in
            `/app/ComfyUI/output/`.

    Returns a workflow dict with a terminal `ShowText|pysssss` node whose
    captured string is the final file path on the server."""
    g = WorkflowGraph()
    staged_path = f"{COMFY_INPUT_DIR}/{staged_subfolder}"

    # MergingVideoByPlenty's output_path is a DIRECTORY that must pre-exist.
    # Use the server's top-level output dir (always present) and rely on the
    # ShowText sink to surface the filename the node picks.
    merged = g.node("MergingVideoByPlenty",
                    video_path=staged_path,
                    output_path=COMFY_OUTPUT_DIR)

    final_path_str = merged[0]
    if audio_filename:
        add = g.node("AddAudio",
                     video_path=final_path_str,
                     audio_from="audio_file",
                     file_path=f"{COMFY_INPUT_DIR}/{audio_filename}",
                     delay_play=0,
                     output_path=COMFY_OUTPUT_DIR)
        final_path_str = add[0]

    # Surface the final path as a captured STRING so the caller can parse
    # it from the execution history and fetch via /view.
    g.node("ShowText|pysssss", text=final_path_str)

    return g.to_dict()


def stage_clips_for_concat(local_paths, subfolder):
    """Upload N local video files into a single comfy-input subfolder with
    zero-padded numeric filename prefixes so alphabetical order matches the
    caller's sequence. Returns the remote names actually used.

    E.g. caller passes [a.mp4, b.mp4, c.mp4] and gets back
    ['001_a.mp4', '002_b.mp4', '003_c.mp4'] stored under
    `/app/ComfyUI/input/<subfolder>/`."""
    from core import upload_image  # lazy: core imports post in some places
    out_names = []
    for i, lp in enumerate(local_paths, 1):
        name = f"{i:03d}_{Path(lp).name}"
        remote = upload_image(str(lp), subfolder=subfolder, upload_as=name)
        out_names.append(remote)
    return out_names


def concat_videos(video_filenames, audio_filename=None, fps=24.0,
                  trim_durations=None, trim_starts=None,
                  filename_prefix="concat", format="mp4", codec="h264"):
    """Stitch N pre-rendered videos end-to-end inside ComfyUI.

    Uses the lighter BatchImagesNode + AudioConcat graph. Per-scene audio
    tracks ARE preserved and chained (drop the separate song overlay — if
    your scenes are ia2v the per-clip audio already IS the song slice,
    stitched = the full song timeline).

    video_filenames: list of video paths (uploaded to ComfyUI input/).
    audio_filename:  optional fallback overlay. Only used if none of the
      clips carry audio. Prefer ia2v-produced clips so this stays unused.
    trim_starts / trim_durations: optional per-clip trim (used with
      transitions). Identity when both are None or match clip duration.

    Heavy memory note: all N clips' frames go through one BatchImagesNode
    in the comfy server. A prior implementation OOM'd at 5 × 896×1664×24fps
    (≈22 GB RSS). If you hit OOM, call this function pairwise from the
    caller (concat 1+2 → A, then A+3 → B, …) — each pass is then only
    2 clips in flight."""
    if not video_filenames:
        raise ValueError("video_filenames must be non-empty")

    g = WorkflowGraph()

    # Per-clip LoadVideo → GetVideoComponents (frames + audio).
    comps = []
    for vf in video_filenames:
        vid = g.node("LoadVideo", file=vf)
        comp = g.node("GetVideoComponents", video=vid[0])
        comps.append(comp)

    # Apply trim iff non-identity params provided.
    def _maybe_trim(images_ref, i):
        start = 0
        if trim_starts and i < len(trim_starts) and trim_starts[i] > 0:
            start = int(round(float(trim_starts[i]) * float(fps)))
        if trim_durations and i < len(trim_durations) and trim_durations[i] > 0:
            n = int(round(float(trim_durations[i]) * float(fps)))
            return g.node("GetImageRangeFromBatch",
                          images=images_ref, start_index=start, num_frames=n)[0]
        if start > 0:
            return g.node("GetImageRangeFromBatch",
                          images=images_ref, start_index=start, num_frames=100000)[0]
        return images_ref

    images_per = [_maybe_trim(c[0], i) for i, c in enumerate(comps)]
    audios_per = [c[1] for c in comps]

    # BatchImagesNode (VHS) uses dynamic inputs named images.imageN.
    if len(images_per) == 1:
        images_out = images_per[0]
    else:
        batch_inputs = {f"images.image{i}": img for i, img in enumerate(images_per)}
        images_out = g.node("BatchImagesNode", **batch_inputs)[0]

    # Chain audio via AudioConcat (binary node: audio1 + audio2 → output).
    audio_out = None
    if audios_per:
        audio_out = audios_per[0]
        for next_audio in audios_per[1:]:
            audio_out = g.node("AudioConcat",
                               direction="after",
                               audio1=audio_out, audio2=next_audio)[0]

    # Fallback to a provided audio file if we somehow ended with nothing.
    if audio_out is None and audio_filename:
        audio_out = g.node("LoadAudio", audio=audio_filename, audioUI="")[0]

    if audio_out is not None:
        created = g.node("CreateVideo", images=images_out,
                         fps=float(fps), audio=audio_out)
    else:
        created = g.node("CreateVideo", images=images_out, fps=float(fps))

    g.node("SaveVideo", video=created[0], filename_prefix=filename_prefix,
           format=format, codec=codec)
    return g.to_dict()
