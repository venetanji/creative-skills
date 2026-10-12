# Experimental native MiniMax H3 FL8 / REF20

For actual prompt composition, read the maintained
[native H3 prompting guide](../comfyui/references/minimax-h3-prompt-guide.md).
Short prompt strings in this profile's CLI examples are transport/admission
fixtures, not recommended generation prompts. Use the complete official
base/I2VA or REF representation before an approved submission.

Bounded source-only slices: `creative_comfy_graph.minimax_h3.build_t2v`
returns a deterministic, pure ComfyUI API graph. The separate skill adapter
`comfyui/scripts/h3.py` owns parsing and real submission/history/download through
existing `core.py`. The new `comfy_graph.py h3-t2v` command dispatches before
permissive legacy parsing or LTX `--model-profile` selection. Existing LTX
signatures, defaults, models, timing, audio and callers are unchanged.

## Exact Recipe

| Role | Curated value |
| --- | --- |
| Transformer | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| FL8step LoRA | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, strength 1 |
| Encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, CLIP type `minimax` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` (not stock FP16) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| Conditioning / joint latent | Native `MiniMaxH3ImageToVideo`, **no optional images** |
| Guidance / schedule | `BasicGuider`, video shift 12 / audio shift 3, Euler / simple, 8 steps, denoise 1 |
| Decode / mux | Same sampler output to `VAEDecode` and `VAEDecodeAudio`; `CreateVideo` 24 fps, 8 bit, generated audio; `SaveVideo` MP4/H.264 |

This is native local H3, not Hunyuan 3D or a partner/API node. FL4step,
reference-family adapters and LTX recipes are not interchangeable.
The stereo joint latent and nominal 32 kHz audio come from the native model /
audio VAE, not from an external soundtrack. An audio stream is not proof of
visual/contact synchronization or prompt fidelity.

## Strict Curated Admission

- Defaults: 864x480, 124 frames at native 24 fps, seed **0**. No random/time seed.
- Pure builder integers must be actual Python ints, never bool, float or string.
  CLI decimal integer tokens convert to ints; JSON values must already be ints.
- Width/height: multiples of 32 in 32..16384, based on captured node schema.
  These registered bounds do **not** certify GPU feasibility or image quality.
- Frames: 124..362, exactly `17k+5` (124, 141, ..., 362). The native implementation
  ordinarily snaps upward; the curated builder rejects off-grid requests before
  transport instead, so requested frames are unchanged. This uses H3's own
  temporal rules, not LTX's `8k+1`. No seconds-to-frames approximation.
- Native source computes video latent T as `((frames-5)//17)*5+2` and audio latent
  T as `round((frames/24)*40)`. Their duration rounding differs; exact decoded
  sample count/alignment needs live inspection. No silent crop/padding promise.
- Seed: 0..2**64-1, including both endpoints, matching captured RandomNoise.
- Prompt: nonblank string without NUL; include desired visual action and sound.
- Prefix: at most 200 characters; slash-separated `[A-Za-z0-9_-]+` literal
  relative segments. Absolute/traversal paths, backslashes, spaces, formatting
  macros and dynamic filename expansion are rejected.
- On `h3-t2v`, images/references remain unsupported. On all routes, supplied
  audio, last frames, reference video/audio, control, negative/CFG, model
  paths, alternate LoRAs, steps, sampler, FPS and quality flags are unsupported.
  Even an explicit apparently default unsupported option is rejected.

The pure builder performs no I/O, backend discovery or GPU/model imports.
Admission is static. It does not certify that these assets can load on the
selected backend. No silent fallback to another family, profile or server.

## CLI And Transport

Use a scoped environment with `creative-comfy-graph` normally installed, not
`PYTHONPATH` or a globally installed/mixed skill script. From this checkout:

```sh
python comfyui/scripts/comfy_graph.py h3-t2v --help
python comfyui/scripts/comfy_graph.py dump h3-t2v --prompt 'One bell strike with decaying brass resonance' --frames 124 --width 864 --height 480 --seed 0
python comfyui/scripts/comfy_graph.py dump h3-t2v --input-json /path/to/spec.json --seed 0
```

Only options: `--prompt`, `--width`, `--height`, `--frames`, `--seed`, `--prefix`,
`--input-json` / `-j`, `--output-dir`, `--timeout`, `--notify-target`,
`--caption-template`, and help. They reuse the current CLI vocabulary; there is
no model-profile or generic graph input. JSON keys may use dashes or underscores;
CLI values override JSON values. Differently spelled JSON keys mapping to the
same normalized option are rejected.
Unknown options, positionals, missing values and malformed input fail locally
with exit 2. Help exits 0. `dump` does no upload, network, probing or GPU work.

Caption templates are validated before transport, even without notifications or
in `dump`, and regardless of CLI/JSON spelling or environment notification target.
The strict subset supports literal text, escaped `{{` / `}}`, the five string
fields `filename`, `workflow`, `id`, `ts`, `prompt`, and optional `!s`, `!r`, `!a`.
Literal string specs may use `<`, `>`, `^` alignment with an optional single fill
character, decimal width, `.precision`, and optional `s`; e.g. `{prompt:>20.10s}`.
Width/precision use one to four decimal digits.
Positional fields, attribute/index access, nested specs, other conversions and
numeric format types are rejected with exit 2 before HTTP or output creation.
Templates are limited to 4096 characters and 32 fields; width/precision are each
at most 1024, with a summed `max(width, precision)` budget of 16384. These bounds
limit requested formatting expansion, not backend-provided string lengths.
Validation uses strings as the transport does (the prompt is its first 100
characters with newlines replaced by spaces); it does not rewrite the template.

Removing `dump` is **real GPU execution**, subject to separate render budget.
Endpoint priority: `COMFY_URL_VIDEO`, `COMFY_URL`, `http://localhost:8188`.
`COMFY_URL_FLUX` is never selected for H3. On Thor explicitly use the verified
video endpoint; Gateway/Thor loopbacks are not interchangeable.

Existing transport submits once, never retries POST, checks terminal successful
history, and requires a downloadable output. Backend rejection, ambiguous POST,
terminal error, missing output and failed download produce a nonzero exit.
Before every submission the shared stdlib `queue_admission.py` helper requires a
fresh `GET /queue` with both list-valued `queue_running` and `queue_pending`
empty. Graph/request serialization finishes first. Busy queues poll every 2s;
read-only connection/5xx failures retry with bounded backoff, never treating a
failed read as empty. Permanent HTTP errors, malformed JSON or missing/null/
non-list fields fail closed. Queue waiting and read-only history retries consume
the same positive integer submission/history timeout (default 600s). Admission
timeout reports `NO_POST` with no submitted prompt ID; accepted-job timeout
retains the ID and says not to resubmit. No POST retry, fallback, OOM recovery or
backend locking is introduced. GET followed by POST is not globally atomic
against external clients. Standalone `comfy_run.py` uses the same guard, including
`--no-wait`, but preserves its separate history budget after an admission budget
of `--timeout` seconds. Copy the shared helper beside both scripts.
Outputs follow existing `OPENCLAW_MEDIA_DIR` / `--output-dir` behavior; this is
not a new job API or cancellation mechanism. Download success is not a decoder
pass or live media certification.

## Retained Provenance And Readiness

Source: upstream FL8step workflow at immutable commit
`02e26d591f7a04d5d1a074c9566d5dd4f22f6225`.
The Oct 2 parent preflight converted it to a 16-node API prototype with explicit
installed int8 video VAE, 864x480 / 124 frames / 24 fps, and bell prompt seed
20261002. This graph is retained unmodified in `tests/fixtures/h3_t2v_prototype.json`:
SHA256 `05a8fdce0a35798151b148f3c82644e6f19521127625c235e1585157c75fc242`.
The selected live native node schemas are retained unmodified in
`tests/fixtures/h3_native_schema.json`:
SHA256 `fb165bf7b747278ad89b9ca6c6c4efd4642aa1ebe28fccbf458746cb66172070`.
Tests assert exact prototype parity, required/optional fields, captured choices /
bounds, link type compatibility, acyclicity, output reachability and no API nodes.

| Scope | Asset-ready | Offline | Live / creative review | Runtime installed | Product exposed |
| --- | --- | --- | --- | --- | --- |
| FL8step text + generated AV | Selected five roles present | Pure graph/admission and executable CLI/transport doubles | Prior native pilot plus two successful native text graphs at 864x480/124f; prior one-strike intent fails; typed text wrapper not live-run | No, scoped test environments only | No Easel/client/CLI integration |
| FL8 I2V / REF20 semantic images | Native schema captured; REF checkpoint listed | Pure recipes/V3 links, atomic admission and real CLI HTTP doubles pass | Six typed I2V and four typed two-image REF20 jobs succeeded in the bounded 2026-10-02 batch at 864x480/124f; native AV fully decoded; creative constraints partly fail | No, scoped test environment only | No |
| FLF / continuation | Some related assets present | Not implemented here | Not certified here | No | No |
| Other control / alternate recipes | Not admitted | Rejected | Not certified | No | No |

Offline transport tests replace only external HTTP; real CLI dispatch, graph,
submission/history/output selection and download logic execute. Text tests verify one
mocked POST to the video endpoint and no uploads/retries. Image tests additionally
verify guarded mock uploads after atomic admission, managed names and one prompt
POST after a fresh empty queue; failures never resubmit. They are **not** live
GPU/render tests. Do not run the legacy `test_all_workflows.py` render sweep as
unit verification. Publication, installation, consumer integration, deployment
and expanded rendering remain separately authorized work.

## Single Native Graph Pilot (2026-10-02)

One local RTX 3090 submission of the retained prototype, seed 20261002, completed
successfully in about 90.6 seconds: prompt ID
`9bc9fbaa-f2d3-404e-a8fd-b159e058805f`. Budget was exactly one POST, zero uploads
and no retries. This exercised the explicit downloaded int8-convrot video VAE,
FL8step recipe and native joint audio/video decode, not the new CLI executable.

Parent independently rechecked successful backend history, graph equality in
the history and MP4 prompt tag, SHA256 and full ffmpeg audio/video decode. The
result is 864x480, 124 frames at 24 fps (5.166667s), with AAC 32 kHz stereo
(5.167s). Near-equal stream duration is not proof of event synchronization.

The intended scene was one visible brass-bell strike around second 2 followed
by ringing decay. Inspection instead finds repeated mallet swings and four
audio-envelope attacks. Material rendering and composition are useful; exact
one-event control and physical grounding fail or remain uncertain. Waveform
attack times and frames are evidence, not an audio audition or certification of
bell timbre, absence of voice/music, or perceptual synchronization.

The exact API graph, unmodified upstream recipe, native schemas, terminal
history, playable clip, frames and audio evidence are retained outside the
repository in the agent workspace under
`media/minimax-h3-exploration-20261002/preflight/` and `bell-pilot/`. Video SHA256:
`0d92139a130b7b1b264859518b44490e0851b542f03738ccb270a3a9a8eaf60c`.

This is one bounded native graph case, not acceptance of every schema-admitted
resolution/frame count, a live CLI caller test, installed runtime support, or
Easel/product exposure. No second render or release was performed.


## Image And Semantic Reference Slices

Usable strict routes are `h3-i2v --image LOCAL.png` and
`h3-r2v --references LOCAL1.png [LOCAL2.jpg]`. All three routes use
`COMFY_URL_VIDEO`, falling back only to `COMFY_URL`; never the Flux endpoint.
These are offline-tested source contracts, not installed or live-certified
capabilities. First live acceptance belongs to the approved batch slots, not
extra pilots.

Pure APIs are `build_i2v(prompt, *, image_filename, ...)` and
`build_r2v(prompt, *, reference_filenames, ref_image_size='match', ...)`.
They accept only safe managed PNG/JPEG basenames, not filesystem paths, backend
output annotations, arbitrary models, or any transport. They retain the same
32-grid bounds, strict integer/uint64 types, seed zero, 17k+5 frame grid,
124..362 range, fixed 24 fps and safe prefix contract as `build_t2v`.
The pure package remains standard-library-only with no Pillow dependency.

I2V is the existing FL8 recipe plus `LoadImage -> MiniMaxH3ImageToVideo.first_frame`.
Native source takes the first loaded image, plain-stretches it to the canvas,
and sets keyframe index 0. It has NO strength parameter and does not inherit
LTX's 8-frame snapping. Last-frame/continuation/multiguide support is not exposed.

R2V uses `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, `res_multistep`,
`simple`, 20 steps, denoise 1, BasicGuider, and **no LoRA or SigmaShift**.
This matches the upstream active `False` branches selecting base model/20 steps;
its optional REF4 branch is neither admitted nor claimed validated. The local
curated video VAE is explicitly the existing int8-convrot VAE (the native
upstream template uses FP16); encoder/audio VAE and joint decode/mux are shared.
Both VAEs are linked to native `MiniMaxH3ReferenceToVideo` so image references
also enter video-VAE reference latents, not just the text encoder.

Admit only 1..2 semantic images in caller order. Prompt tags include spaces:
**`<Picture 1>`**, **`<Picture 2>`**, matching the order of `--references` or the
JSON array. The wrapper does not silently rewrite or auto-number prompt prose.
`--ref-image-size match|max` is R2V-only. `match` downscales references to the
canvas pixel area; `max` retains up to a 2048px short edge and can be much slower.
Neither tag fidelity nor identity preservation is certified by graph admission.

### Admission And Offline Dump

The image CLI alone requires explicitly provisioned `Pillow>=10` in its
executing environment (`comfyui/requirements-h3.txt`). It never installs on
import and has no sibling/global dependency fallback. Text/help/pure imports
do not require Pillow. `dump` of an image route validates local images but
performs **zero HTTP, uploads, probing, POST, or GPU/model execution**.
Its complete graph contains stable `h3_INDEX_SHA256.png|jpg` managed placeholders;
it never puts local paths into backend LoadImage nodes.

All option types, graph fields, caption templates and ALL local image files
must pass before the first upload. Input files must have a PNG/JPG/JPEG suffix,
matching decoded PNG/JPEG format, one frame only, at most 32 MiB each,
1..8192 pixels per edge, and at most 16 Mi pixels. Full Pillow `load()` is
required; truncated/corrupt images and decompression-bomb warnings fail closed.
These are caller resource bounds, not model or GPU feasibility certification.
Admitted bytes are hashed and staged locally, so changes to the original source
between uploads cannot replace already-validated contents. No image conversion,
external decoder, shell, or model execution is used.

The adapter reuses `core.upload_image` only after atomic local admission, with
managed hash basenames at the selected video endpoint. Only the expected basename
or a numeric collision suffix is accepted (max 200 characters); paths, output
annotations or arbitrary backend-selected names fail. Uploads have no retries.
Upload/staging/name-validation failures report **NO_POST** and warn that previously
uploaded input assets may remain. The existing core then requires a fresh empty
0-running/0-pending queue immediately before exactly one `/prompt` POST. Admission
and accepted-job failures retain existing no-resubmission transport behavior.

```sh
python comfyui/scripts/comfy_graph.py dump h3-i2v --image /local/anchor.png --prompt 'The person walks toward the door. Audio: soft footsteps.' --frames 124 --seed 0
python comfyui/scripts/comfy_graph.py dump h3-r2v --references /local/person.png /local/prop.jpg --ref-image-size match --prompt 'The person in <Picture 1> holds the object in <Picture 2>. Audio: quiet room tone.' --frames 124 --seed 0
```

Remove `dump` only for an authorized media job. `--image` is I2V-only;
`--references` (JSON array) and `--ref-image-size` are R2V-only. JSON scalar
references, booleans, unsupported audio/video/control/custom model/sampler/
negative/strength options (even `--strength 1`) fail locally. Common options and
CLI-over-JSON overrides are unchanged. Image routes also preflight captions,
including in dump and without notifications.

### Native API Serialization Evidence

`MiniMaxH3ReferenceToVideo` uses deployed V3 Autogrow, not a nested list of
NodeRefs: API inputs are direct links `ref_images.ref_image_0: ['16', 0]` and
`ref_images.ref_image_1: ['17', 0]`. Captured deployed `/object_info` defines
`COMFY_AUTOGROW_V3`, prefix `ref_image_`, max 9; this curated route narrows to 2.
Import-free extracted deployed `_io.py` functions prove schema expansion and
reconstruction into the ordered execution argument `ref_images` dictionary.
No torch/ComfyUI imports, validator POSTs, or node/model executions are needed.
Fixtures retain native links/types, upstream false-switch links and separate
I2V/REF API recipes. LoadImage's mutable filename inventory is redacted to fixture
basenames; uploaded managed names are not required to preexist offline.

Evidence SHA256:
- Native `nodes_minimax_h3.py`: `108ceff11e5a5c9a34a57bbc380447cf9941beb673a1bd421abc818de54ccd5c`
- Native `video_minimax_h3_r2v.json`: `14b30659a057547e02bdd4bbbdda3f8670aa6d7d81d1d8d99c4f9ad1e2eabc44`
- Deployed V3 `_io.py`: `e697ed00732deffb46d7a017d82dae48ee2504a8d356f911a1447ec6146294f7`

This proves serialization/topology and offline transport admission, not live
REF20 loading, timing, generated audio, identity fidelity, or decoder success.

## Supplied Audio: IA2V (12 October 2026)

`build_ia2v(prompt, image_filename='anchor.png', audio_filename='track.wav', ...)`
retains the FL8 first-frame recipe and adds `LoadAudio`, `TrimAudioDuration`
and native `MiniMaxH3AddGuide` at frame zero. The guide consumes the H3 audio
VAE and original AV latent; its conditioning feeds `BasicGuider`. H3 handles
the frozen temporal audio anchor internally. Do not substitute LTX AV latent
or noise-mask nodes. The existing FL8 eight-step schedule is unchanged.

The trimmed source feeds `CreateVideo` directly; generated `VAEDecodeAudio`
is omitted. This is audio-conditioned generation, not soundtrack-only muxing,
and avoids replacing a supplied song with the model's reconstructed audio.
Native guide behavior comes from the pinned ComfyUI H3 node source cited above.

```sh
python comfyui/scripts/comfy_graph.py dump h3-ia2v \
  --image /local/anchor.png --audio /local/track.mp3 \
  --prompt 'A person sings to camera' --frames 124 --seed 0
```

The CLI accepts local WAV/MP3/FLAC/Ogg/M4A audio up to 32 MiB, with at most
eight channels and 192 kHz source rate. Both media files and all options are
validated before uploads. ffprobe/ffmpeg run locally with network protocols
and playlist formats disabled and 15-second subprocess limits; the first
audio stream is normalized to 48 kHz stereo PCM WAV. It must cover exactly
`frames/24` seconds; longer tracks are trimmed from zero, short tracks reject.
The normalized bytes are staged, hashed and uploaded under managed basenames.
Pillow and ffmpeg/ffprobe must be provisioned separately from the pure package.
Dump validates/decodes local files but makes no backend request. Upload failure
never triggers a generation POST, and prompt submissions are never retried.

This path has offline graph, real codec/CLI admission and guarded HTTP tests.
There is no new live GPU or perceptual lip-sync acceptance; the generated-AV
pilots elsewhere in this document do not validate this source-audio extension.

## Bounded review acceptance, 2026-10-02

The twelve-slot review completed two standalone text graphs, six typed FL8
first-frame clips and four typed two-image REF20 clips. Each has 124 decoded
frames at 24 fps, 864x480 video and generated 32 kHz stereo audio; terminal
success and full AV decoding were checked. Exactly 12 prompt POSTs, no rerolls
or fallback models; running and pending queues were both empty before each
submission. Two Qwen scene anchors were prepared separately. Text clips used
the standalone graph runner, not the typed text CLI. Matched controls 03/06
are pixel-identical, not independent visual draws. Chaining used actual prior
frame 123; generated frame zero is not certified pixel-identical or seamless.

Validation covers this shape, length and selected assets only. Some grounded
bell turns lift the bell. Semantic references can carry the identity photo's
background despite a separate scene reference; a wider portrait request
produced a prop close-up. Technical success does not imply creative acceptance
or perceptual sound/action synchronization. Full native originals and case
receipts remain outside the repo at
`media/minimax-h3-review-batch-20261002/`. No global installation or product
release occurred. The 62-second review reel retains clip audio content in
order but resamples it to 48 kHz and reencodes AAC; native originals stay 32 kHz.
