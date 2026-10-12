# Model-specific video prompting

Maintained 2026-10-02. This is a source-backed writing reference, not a render
scheduler, model installer or permission to change a tested sampling recipe.

## Choose the actual model and conditioning mode

| Model / mode | Read before composing | Do not substitute |
| --- | --- | --- |
| LTX-2.5 | [2.5 guide](ltx-25-prompt-guide.md) | H3 fields or the legacy 2.3 cheatsheet |
| Native H3 base / FL8 first frame | [H3 guide](minimax-h3-prompt-guide.md), base section | Hailuo API advice or REF field grammar |
| Native H3 REF20 semantic images | [H3 guide](minimax-h3-prompt-guide.md), reference section | Ordinary I2V or an LTX Ingredients sheet |
| Legacy LTX-2.3 | [2.3 cheatsheet](ltx-prompt-guide.md) | An implied profile upgrade |
| Shot anchors / layout | Model's image guidance plus the anchor workflow | A neutral identity sheet as a finished scene anchor |

Official prompting advice, upstream node contracts, local execution evidence,
and production proposals are different evidence classes. The linked guides
label those boundaries. A public model capability does not create a local
CLI/Easel feature; a successful decode does not prove identity or control fidelity.

## Before generation and review

1. State the shot's expressive purpose, visible subjects, framing and motion.
2. Read the exact model/mode guide and the selected adapter's own card. Match
   reference labels to real admitted inputs; never invent conditioning with text.
3. Inspect the actual anchor/reference at target aspect. Check proportions,
   perspective, crop, scene lighting and unwanted reference-background content.
4. Keep a bounded submission budget and check both backend queue lists empty
   immediately before each POST. No automatic reroll or model fallback.
5. Review native frames and sound against the intention. Separate technical
   success, appearance, camera/control fidelity, audio synchronization and
   the user's acceptance. Correct a bad anchor before asking I2V to contradict it.

## Evidence and maintenance

[Source records](model-prompting-sources.json) retain public URLs, immutable
revisions when available, access timestamps and response SHA256s. The full
reviewed snapshots are outside the repository in the retained task evidence
`media/model-prompt-guides-20261002/{ltx25,h3}/`. Do not infer a reliable release
date from contradictory page/search metadata; use revisions and access dates.

Refresh the affected guide when its checkpoint, node/template, adapter recipe,
official guide, or a meaningful creative result changes. Re-fetch the source,
compare to the recorded snapshot/revision, and preserve unaffected advice.
Update mode-specific examples, capability boundaries and source records together.
Resolve syntax disagreements with the model maker; adapter settings with the
selected adapter maker and actual tested graph. Keep empirical observations
separate, with a retained evidence pointer and scope. Do not generalize one case.
Validate links, examples/field order and source hashes, and review against the
selected graph. New renders require their own budget; no scheduler is implied.
Publication, merge, installation and consumer deployment retain their own scope.
