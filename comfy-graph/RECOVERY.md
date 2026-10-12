# Canonical graph recovery, 12 October 2026

The active Creative Skills repository's `comfy-graph/` directory is the
canonical home of `creative_comfy_graph`. Do not import archived `/tmp`
checkouts or depend on the retired `agentic-media` workspace at runtime.
Skill adapters import the explicitly installed graph distribution; recovery
does not install skills, replace a running environment, publish a release,
or submit GPU jobs.

## Recovered sources

- Archived Creative Skills main at `0e208b2312b33fbf31b95fb15e25273195baa9be`:
  standalone graph extraction, characterization/artifact tests, strict LTX
  profile adapter, LTX-2.5 source-audio/frame-exact continuation and first/last
  profiles, plus associated caller/timeline fixes.
- Uncommitted `feat/minimax-h3-t2va-20261002` worktree based on that revision:
  native H3 recipes, prototypes/schema tests, strict H3 CLI/image admission,
  queue admission, prompting references and packaging support. Unrelated
  pending storyboard review work remains in the archive.
- Current public main: preserve SAM 3D Body support, newer transport changes,
  and unrelated local owner edits. SAM imports are lazy so copied non-SAM
  graph entrypoints do not require that optional adapter.

Historical snapshots remain unchanged. This is a content recovery into the
active main worktree, not a claim that orphaned commits were already published
or that every historical workflow has live GPU validation.

## New source-audio extension

`minimax_h3.build_ia2v` and `comfy_graph.py h3-ia2v` combine FL8 first-frame
conditioning with native H3 audio guides, then mux the trimmed source track.
The CLI validates and normalizes local audio before any upload. LTX-2.5's
source-audio profile is retained, not silently made the legacy CLI's default.
See the profile documents for limits and evidence.

Easel independently exposes multipart `source_audio` for LTX-2.5 and H3,
without importing this package or temporary source paths at runtime. Both
repositories distinguish graph-contract tests, runtime node availability,
and live/visual acceptance; recovery adds no live IA2V quality evidence.
