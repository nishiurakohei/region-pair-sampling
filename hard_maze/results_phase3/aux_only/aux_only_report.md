# Phase 3-A — Random-pair Connectivity Auxiliary-only

## Decision

Best auxiliary accuracy: **CNN 0.747**. Decision: **PROCEED to Phase 3-B**.

| model | accuracy | AUROC | BCE | Brier |
|---|---:|---:|---:|---:|
| CNN | 0.747 ± 0.004 | 0.825 | 0.510 | 0.170 |
| RESNET | 0.687 ± 0.008 | 0.757 | 0.588 | 0.201 |

Pairs exclude start/goal and identical nodes, use grid distance ≥5, and are exactly balanced on fixed validation/test sets. Figure 1 and Figure 2 show distance-stratified performance. The threshold for proceeding was auxiliary accuracy >0.65.
