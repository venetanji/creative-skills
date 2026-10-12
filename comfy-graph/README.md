# Creative Comfy Graph

An independently installable, MIT-licensed Python >=3.11 library with zero
third-party runtime dependencies. Distribution: `creative-comfy-graph` 0.1.0;
import namespace: `creative_comfy_graph`. These are the canonical graph sources
for this repository's skill adapters; see [RECOVERY.md](RECOVERY.md) for the
archived history and pending H3 work recovered onto main. No external release
or deployment is implied. The package contains:

| Import | Contents |
|---|---|
| `creative_comfy_graph` | `NodeRef`, `WorkflowGraph`, `__version__` |
| `creative_comfy_graph.flux2` | Flux2 text-to-image, reference edits, multi-prompt and angle builders |
| `creative_comfy_graph.tts` | Qwen speech and voice-clone builders |
| `creative_comfy_graph.post` | Stem/transcription and concatenation graph builders |
| `creative_comfy_graph.minimax_h3` | Experimental FL8 text/first-frame and REF20 semantic-image native generated AV builders (offline-only) |
| `creative_comfy_graph.ltx_profiles` | Explicit LTX-2.3/LTX-2.5 profiles with frame-exact source-audio conditioning and mux |
| `creative_comfy_graph.ltx2` | Existing LTX-2.3 video, guide, transition, continuation and last-frame builders |

Builders return ComfyUI graph dictionaries; they do not execute them or validate
server/model availability. No Torch, ComfyUI installation, weights, OpenClaw or
Easel is needed to import/build graphs. Transport, uploads, polling, downloads,
CLI orchestration and local clip staging remain in the legacy skill adapters.
In particular, `stage_clips_for_concat` is not a library API. Server-side filename
and model strings describe graph inputs, not files opened by the package.

```python
from creative_comfy_graph import WorkflowGraph
from creative_comfy_graph.flux2 import flux2_text_to_image

graph = WorkflowGraph()
source = graph.node("ExampleSource", value=42)
graph.node("ExampleSink", source=source[0])
print(graph.to_json())
workflow = flux2_text_to_image("fixture", seed=42)
```

Node IDs are zero-based strings. Only direct NodeRef inputs become links; nested
references remain untouched. `to_dict()` copies the outer dictionary only.
Legacy `core.NodeRef` and `core.WorkflowGraph` re-export the same classes.
Legacy family modules retain builder signatures, constants and aliases.

## Explicit Environment Provisioning

Install uv first and use an available Python 3.11 or 3.12 interpreter. Nothing
installs dependencies on import. For checkout development, substitute your paths:

```sh
ROOT=/path/to/creative-skills
ENV=/tmp/comfy-env
uv venv --python 3.11 "$ENV"
uv pip install --python "$ENV/bin/python" --link-mode=copy -e "$ROOT/comfy-graph"
```

An editable install intentionally depends on that checkout. For copied skills,
use a wheel instead: create the environment with `uv venv` as above, then build
and install the local distribution into the interpreter executing those scripts:

```sh
DIST=/tmp/comfy-dist
SCRIPTS=/path/to/copied/comfyui/scripts
uv build "$ROOT/comfy-graph" --out-dir "$DIST"
uv pip install --python "$ENV/bin/python" --link-mode=copy --no-deps \
  "$DIST/creative_comfy_graph-0.1.0-py3-none-any.whl"
export PATH="$ENV/bin:$PATH"
python3 -c 'import creative_comfy_graph as p; print(p.__version__, p.__file__)'
python3 "$SCRIPTS/comfy_graph.py"
```

The no-argument invocation prints usage and exits 1 (expected, not a failed
installation). Copy the tracked skill scripts together, not just the CLI; no
package source needs to be copied. Copy mode is scoped to each install to avoid
cross-filesystem uv cache hardlink fallback, not a global uv setting.

Set PATH explicitly in the parent environment and inherit it into subprocesses.
A uv interpreter selection or shebang alone does not provision bare `python3`
children. This smoke check exercises an actual uv parent and plain Python child:

```sh
uv run --no-project --python "$ENV/bin/python" python -c \
  'import subprocess; subprocess.run(["python3", "-c", "import creative_comfy_graph as p; print(p.__file__)"], check=True)'
```

Do not set PYTHONPATH to package source. There is no sibling-checkout discovery,
installed-skill fallback or auto-install. Missing installation fails before
transport work with:

```text
creative-comfy-graph is required; install it into the Python environment running this script.
```

## Offline Verification

Provision pytest separately for development; it is not a runtime dependency.
The repository is needed by the outer harness to inventory tracked legacy scripts.
Use absolute paths and a new work directory outside the checkout:

```sh
FAMILIES=primitives,flux2,tts,post,ltx2
uv pip install --python "$ENV/bin/python" --link-mode=copy pytest
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$ENV/bin/python" -m pytest -q -p no:cacheprovider "$ROOT/comfy-graph/tests" \
  --legacy-dir "$ROOT/comfyui/scripts" --expected-families "$FAMILIES"
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$ENV/bin/python" "$ROOT/comfy-graph/tests/tools/accept_artifacts.py" \
  --project "$ROOT/comfy-graph" --legacy-dir "$ROOT/comfyui/scripts" \
  --work-dir /tmp/comfy-artifacts \
  --expected-families "$FAMILIES" --python 3.11 --python 3.12
```

The harness builds a wheel/sdist and a wheel rebuilt from that sdist, checks
archive allowlists, MIT license, metadata and absent runtime dependencies, then
installs each wheel into fresh actual Python 3.11/3.12 environments outside the
checkout. It copies tests/fixtures and tracked legacy scripts (excluding the live
workflow sweep), runs from an unrelated empty cwd with normal site-packages
imports, and checks all 197 characterized cases across 22 builders against
immutable goldens through both legacy adapters and the extracted families.

Each runtime checks actual guarded CLI `__main__` usage and t2i dump, original
function exports/module aliases and uv-parent/plain-python3-child identity.
Separate empty interpreters verify actionable missing-package failure before
network, subprocess or media work. Inner tests exclude `artifact_driver` to
prevent recursion and validate supplied archives instead of rebuilding. The
harness writes commands/results to stdout and artifact SHA-256 hashes, versions,
package locations and test counts to `<work-dir>/evidence.json`. Development
runs skip two artifact-only checks; all four wheel runs execute them without skips.
Build/test dependency acquisition may require network; graph/runtime fixtures
block network, subprocess and media boundaries. No ComfyUI server, GPU, model,
media probing or upload is required for this acceptance gate.

## Compatibility And Limitations

This is an extraction, not a deterministic API redesign: ambient `time.time()`,
seed-None/seed-zero inconsistencies, `LTX_SAGE_ATTENTION`, warnings and ignored
extra LTX keywords remain unchanged. So do frame rounding, crop/upscale order
and audio semantics: first/last, multi-guide and continuation crop after upscale;
transition uses integer-FPS audio conditioning and muxes VAE audio, unlike the
other source-audio workflows. Graph parity is not proof of a successful render
or supplied-audio fidelity.

The guarded t2i dump fixture does not make arbitrary legacy help/dump handlers
safe; existing paths may upload or submit. There is no LTX-2.5 addition, live GPU
validation, consumer migration or deployment in this milestone. The local build
is not a published immutable dependency pin or proof of a clean consumer Docker
build; publication and consumer integration require separate work.

## Experimental Curated LTX Profiles

See the repository `comfy-graph/LTX-PROFILES.md` for explicit 2.3/2.5 loaders,
frame-based continuation/source-audio semantics, CLI admission and measured
live capability limits. This does not promote a new default or fix legacy callers.

The curated `ltx_profiles.build_flf2v` and selected-profile CLI `flf2v` accept
two independently weighted single-image endpoints with optional source audio.
These combinations are OFFLINE-validated only; see [LTX-PROFILES.md](LTX-PROFILES.md)
for final dimensions, endpoint timing, upload preflight and live acceptance limits.

## Experimental Native MiniMax H3

`minimax_h3.build_t2v` and `comfy_graph.py h3-t2v` preserve the native FL
eight-step graph with generated stereo audio. See [H3-PROFILE.md](H3-PROFILE.md)
for exact roles, strict admission, provenance, safe preview, and readiness.
This does not change any LTX default, install a skill, or expose H3 in Easel.
The legacy `--expected-families` characterization selector remains unchanged;
H3 has its own retained native prototype/schema and executable CLI tests, also
run by the all-family artifact harness.


H3 image routes: `h3-i2v --image LOCAL.png` (FL8 first-frame) and
`h3-r2v --references LOCAL1.png [LOCAL2.jpg] --ref-image-size match` (REF20).
The pure `minimax_h3.build_i2v`/`build_r2v` APIs use managed input basenames and
remain dependency-free. Only image-bearing CLI admission requires separately
provisioned Pillow; all files/options/captions validate before any upload.
Image `dump` is fully offline with stable hash placeholders. Exact native
V3 dotted-port/schema proof, curated recipes and remaining live acceptance are
in [H3-PROFILE.md](H3-PROFILE.md).

`minimax_h3.build_ia2v(..., image_filename='anchor.png', audio_filename='track.wav')`
adds native audio-guide conditioning to FL8 and muxes the trimmed source
soundtrack. The strict CLI is `h3-ia2v --image LOCAL.png --audio LOCAL.mp3`.
It validates and normalizes both files before any upload; ffmpeg/ffprobe are
required in addition to Pillow. Source-audio graph/CLI contracts are tested
offline; existing H3 generated-AV pilots do not certify this path or lip-sync.
