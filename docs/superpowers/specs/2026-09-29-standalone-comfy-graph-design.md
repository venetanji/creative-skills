# Standalone Comfy Graph: focused architecture amendment

Date: 2026-09-29.
Status: approved by Gio on 2026-09-29 at 17:48 UTC; task-level M1 plan written and self-reviewed, awaiting plan review; product implementation not started.
Roadmap: `ROADMAP.md`, milestones M0-M2.

## Intent and retained decisions

Make Comfy Graph a standalone Python library that creative skills and Easel can install through uv, rather than making application code depend on an OpenClaw skill directory. Simplify ownership and imports without rewriting working workflows.

This amends the package boundary in the earlier approved LTX design, not its product direction. Retain local ComfyUI only for LTX work, explicit 2.3/2.5 profiles, uploaded audio and keyframes, no silent fallback, and 2.5 default promotion only after validation. Studio/Media MCP and Easel CLI video delivery remain roadmap milestones. No new remote repository, publication, deployment or live render is included in the extraction.

## Recommendation and alternatives

**Recommended:** add `comfy-graph/` at the root of `venetanji/creative-skills`, outside every skill directory. It has its own `pyproject.toml`, source tree, license, tests, wheel and sdist. Consumers install the distribution; they do not add the repository or a skill directory to their import path. Standalone packaging and repository ownership are separate decisions.

Use provisional distribution `creative-comfy-graph` and import namespace `creative_comfy_graph`. Avoid `comfy_graph`, which collides with the existing `comfy_graph.py` script on normal script import paths. Check registry availability before any publication; no package name is claimed by this document.

A separate `venetanji/comfy-graph` repository would provide release separation but adds creation/publication and cross-repository migration dependencies immediately. It can follow later without changing the import namespace. The old `comfyui/workflows/` location remains technically installable but puts the library inside the skill and does not match the new ownership direction.

## Verified starting point

- Creative Skills main/local clean reference: `741bab9bb1da495e96949286695fccfd3ec931ff`.
- Easel main/local HEAD: `dbdb484493323e7877bf4b92380387e0b9f37a59`, with owner changes and video drafts; read-only for this milestone.
- Historical design/plan branch: `0a8cc0e`, based on older Creative Skills `19169e0`; do not build the refactor on that branch.
- Installed copies are mixed. Flux/TTS/post match main at review; core/LTX/CLI do not. Do not extract from installed copies.
- Relevant remote branches must be checked before implementation. The architecture review established `806c1ee` (pass-two IC-LoRA) and `d64ae9f` (media directory) as ancestors of current main. The plan-authoring review also confirmed `refactor/strip-storyworld-from-comfyui` at `eccd415` is an ancestor of main (66 commits behind, no branch-only diff); no cherry-pick is needed.

Use isolated implementation work from freshly rechecked main. Preserve dirty source trees, installed skills and other agents' worktrees.

## Package boundary

Target layout after incremental extraction:

```text
comfy-graph/
  pyproject.toml
  README.md
  LICENSE
  src/creative_comfy_graph/
    __init__.py
    graph.py
    flux2.py
    ltx2.py
    tts.py
    post.py
  tests/
```

`__init__.py` stays narrow: version and graph primitives. Family modules retain existing builder names rather than introducing one large generic request object.

Dependency direction:

```text
skill CLI and compatibility adapters -> graph library
Easel service adapters               -> graph library
family builders                      -> graph primitives

skill CLI -> existing transport and OpenClaw output adapter
Easel     -> its own asynchronous ComfyClient
```

The library does not import `core`, OpenClaw, Easel, skill discovery, HTTP/upload helpers or CLI orchestration. No Torch, ComfyUI server or model weights are Python package dependencies. Existing MIT license and attribution travel with the source; this does not license external models or custom nodes.

Keep upload/submission/polling/download, auth, job persistence, admission and media mirroring outside the graph library. `stage_clips_for_concat` remains a transport adapter. Graph nodes may describe server paths without opening local files.

## Migration and compatibility

**First slice: primitives, characterization and packaging only.** Move `NodeRef` and `WorkflowGraph`; re-export the exact same classes from legacy `core.py`. Retain mutable `core.BASE` and all transport functions. No builder rewrite, CLI rewrite, Easel import swap or model change.

**Following slices:** extract Flux/TTS, then graph-only post functions, then LTX. Preserve family interfaces and characterize complete graphs before moving code. Legacy modules become explicit re-exports or adapters, not a second implementation. Do not delete compatibility imports from `comfy_graph.py` that callers may use.

Keep Creative primitive behavior: zero-based string IDs, indexed references returning `NodeRef`, direct-reference conversion, insertion order, shallow `to_dict()` copy and JSON formatting. Do not substitute Easel's similar-looking primitive; it has different attributes, indexing and numbering.

Preserve existing negative prompts, fast/two-pass modes, model overrides, camera/IC LoRAs, references, strengths, transition/continuation parameters, keyword compatibility, filenames and output markers. Preserve exact frame rounding, guide cropping, spatial upscale and source-audio trim/mux semantics.

Easel migration is separate: its image builders have prompt-list, batch, seed-zero and reference-size semantics not reproduced by a simple alias to Creative Skills. Add consumer-parity tests and deliberately extend/adapt the shared API before removing vendored code.

## Purity without hidden behavior changes

Graph primitives are pure immediately. Existing builders perform no transport, but some derive omitted seeds from time, inspect attention environment settings or print warnings. Mechanical extraction retains those defaults temporarily and labels them; it must not be called fully deterministic just because files moved.

A following boundary-cleanup slice moves ambient resolution into legacy/CLI adapters and passes explicit seed/configuration to deterministic internals. Fixed explicit inputs must then produce the same graph independent of time/environment. Preserve legacy defaults through adapters. Do not bury seed-zero fixes, stricter validation, warning policy or swallowed-keyword changes in extraction commits.

## uv and launcher contract

Use an independent build backend with explicit src-package selection and zero third-party runtime dependencies initially. Proposed support floor is Python 3.11, matching Easel; actual 3.11 and 3.12 executions are acceptance gates, not inferred from AST parsing. Preserve license/metadata in both wheel and sdist.

For development, uv can install the package editable into an isolated environment. For acceptance, build a wheel and install only that artifact into a fresh uv-managed environment outside the checkout, with user-site disabled and PYTHONPATH unset. Rebuild a wheel from the sdist too.

**Supported copied-skill execution requires the dependency to be provisioned in the runtime environment.** Keep plain `python3 script.py` compatible when that interpreter has the package. For uv-managed execution, the parent and its `python3` subprocesses must resolve the same environment; test interpreter paths and inherited PATH explicitly. A PEP 723 shebang alone is insufficient because current callers launch `python3 <script>` and bypass it.

Do not auto-install from an import, use `sys.path` to discover a sibling repository, or silently fall back to installed OpenClaw scripts. Missing dependencies fail clearly before upload or submission. Installing/replacing the user's active skill environment is a separate rollout action; local copied-layout fixtures prove compatibility first.

After authorized publication, consumers can use an immutable reachable Git revision with `#subdirectory=comfy-graph`, or a versioned wheel/artifact with verified hash. Local Easel tests may use an explicit temporary wheel/path override; release locks must not require a private absolute path or an unpublished Git revision. Clean frozen Docker builds are a later gate after authorized dependency publication, not claimed by local wheel tests.

## Separate safety and model changes

Immediately after the extraction gates, isolate these intentional fixes with their own tests:

- All help forms return before uploads, file probing, subprocesses, `/free`, submission or download.
- Dump builds graphs without upload/network/subprocess/output writes; explicitly requested JSON input reads can remain allowed. Continuation needs explicit metadata requirements rather than hidden ffprobe.
- Rejected validation submissions expose the actual error and are not retried blindly. Ambiguous transport failure must not create duplicate accepted jobs.
- Installed-helper precedence and environment routing become explicit rather than mixing checkout and installed code.

Do not run current unsafe help/dump against live inputs to test the extraction. LTX-2.5 profiles, capability validation and defaults follow in M3; they are not evidence of behavior-preserving extraction.

## Review and acceptance gates

For each slice, one implementer owns the isolated source changes, a fresh reviewer examines its immutable diff, and the parent verifies evidence and routes fixes before advancing. Parallelize only independent work; do not assign two writers to the same builders.

First-slice requirements:

1. Primitive attributes, indexing, repr, numbering, link conversion, copy semantics and serialization match baseline, including existing edge cases.
2. Package and legacy imports have identical class identity and interoperable references.
3. Characterization graphs with controlled seeds/configuration remain equal for every public Flux/LTX/TTS/post builder; include refined/fast, guides/audio, identity/LoRA, transition and continuation paths. Do not replace equivalence with node counts.
4. Library import and graph construction do not contact the network, open client media, invoke subprocesses or discover OpenClaw. Block those boundaries in offline tests without disabling ordinary Python import loading.
5. Fresh wheel-only imports resolve to site-packages from an unrelated cwd, without source-path assistance. Copied skill layout works with the declared dependency and supported runtime, and missing-dependency behavior is tested.
6. Declared minimum Python 3.11 and Easel Docker's Python 3.12 pass. Wheel/sdist include only intended code/metadata, not media, secrets, caches, models or skill directories.
7. Source checkouts remain unchanged; report exact base/head revisions, test commands, counts, artifacts and limitations.

No GPU test is necessary to establish primitive extraction equivalence. The broad `test_all_workflows.py` live/OOM sweep is not a unit suite.

## Decision recorded

Gio approved the standalone package at `creative-skills/comfy-graph/`, with a primitives-first compatibility-preserving extraction and uv-managed dependency provisioning, on 2026-09-29 at 17:48 UTC ("approved. do it"). Subagent execution was already selected. This settles the new boundary; it does not reopen the previously agreed LTX/video product direction or authorize publication. The next artifact is the task-level implementation plan for the supervised refactor slices.
