# Head-to-head: HC HV-acceptance vs incremental vs crossbreed

- Variants: hc9, hc12, hc15
- Seeds: [42]
- Budgets (s): small-graph=4, medium-graph=4
- Score = official −HV (higher = closer to 0 = better). Re-scored from the written submission, not the live print.
- **Wall-clock-budgeted: hardware-dependent. Quote only reference-workstation runs.**

## Mean ± stdev per variant

| problem | hc9 | hc12 | hc15 | best |
|---|---:|---:|---:|---|
| small-graph | -1,814,925 ± 0 | -1,814,515 ± 0 | -1,814,976 ± 0 | **hc12** |
| medium-graph | -1,607,963 ± 0 | -1,603,494 ± 0 | -1,608,632 ± 0 | **hc12** |

## hc15 deltas vs parents (positive = hc15 better)

| problem | hc15 − hc9 | hc15 − hc12 |
|---|---:|---:|
| small-graph | -51 | -461 |
| medium-graph | -669 | -5,138 |

> Reading: hc15 carries hc9's HV-improvement acceptance on hc12's incremental evaluator. A positive delta on the sparse instances would say the HV-aligned acceptance survives the switch to the cheaper evaluator; a positive delta on the dense instance would say the extra moves-per-second outweigh hc9's more frequent cache rebuilds.
