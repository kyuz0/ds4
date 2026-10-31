# Experiment index

Historical index, not a live ledger. No maintenance is required.
Use newer session results for current status; retrieve history only when needed.

| Candidate | Decision | Evidence |
|---|---|---|
| INDEXER_HEAD_WIDE | Active opt-in, validation pending | findings/profile-append-4k-context-kernels-20261030-231900/ |
| COMPACT_JLIMIT | Performance closed: 0.06% within noise; seeding safety unresolved | bench-compact-jlimit-clean-repeat-ab.sh; search JLIMIT in RESULTS.md |
| Indexer keys4 / keys16 | Rejected; new rationale needed to repeat | bench-indexer-keys4-ab.sh; bench-indexer-keys16-ab.sh; NOTE.md |
| Selective suffix staging | Promoted for supported 1–512 routes; 1024 whole-layer | NOTE.md final selective-stage section |
| All-resident routed MoE | Promoted; redundant compaction removed | NOTE.md all-resident section |
| Attention low-A M256 / output-B pad8 | Promoted | NOTE.md matching sections |
| Async selected readback | Rejected: repeatable short-append regression | NOTE.md async selected section |
| Single-row WMMA / blocks4-v2 | Rejected: regression or logits drift | NOTE.md single-row sections |
