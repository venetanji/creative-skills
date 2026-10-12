# Explicit LTX profile route

See `../drama-video/PROFILED-VIDEO.md` for the shared frame/audio protocol,
installation requirements, accepted fields and limits. The music caller uses
`scenes` and full `song.mp3`, persists `scenes/NNN-label.timeline.json`, and
assembles with one continuous master through production frame-span assembly.
Song variants receive the same frame spans and each get their own master mux
and `final_vN.timeline.json` verification metadata.

Use `video.model_profile: ltx-2.5` or `ltx-2.3`; omission does not change the
legacy route. Image/generated-audio paths are offline-tested, not independently
GPU-validated. LoRAs, transitions, multiguide, ingredients, separate lipsync
tracks and anchor generation are rejected on the explicit profile route rather
than silently lost. `all` skips the unsupported transition pass when profiled.
Keep the sibling drama helper installed alongside this skill; do not mix helper
revisions. Actual caller assembly of existing real A/B clips has been validated;
this is not a claim of a new end-to-end caller GPU render or seamless motion.

Profile `all` uses the same exact frame/master coverage preflight as rendering,
not the legacy longest-variant/freeze-tail duration heuristic. A longer master
is valid: assembly trims it to retained visible time. String durations use the
shared rational planner. Unused variants do not affect generation admission;
when variant exports are requested by assembly, each mux is validated against
its own first audio stream. Known short selected masters fail before additional
paid song-generation work. Restart metadata uses the shared generation-input
fingerprint; rerender affected clips and descendants after changing inputs.
