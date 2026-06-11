# Multi-seed analysis: official instances

- Variants compared: hc5, hc9, hc11, hc14
- Statistical baseline: hc9
- Significance test: Wilcoxon signed-rank (scipy)

## 1. Per-cell summary statistics

| variant | problem | n | mean | median | min | max | σ |
|---|---|---:|---:|---:|---:|---:|---:|
| hc5 | small-graph | 12 | -1,813,936 | -1,813,520 | -1,814,961 | -1,813,206 | 727 |
| hc9 | small-graph | 12 | -1,814,574 | -1,814,448 | -1,816,554 | -1,813,427 | 978 |
| hc11 | small-graph | 12 | -1,813,887 | -1,813,462 | -1,814,911 | -1,813,210 | 719 |
| hc14 | small-graph | 12 | -1,813,743 | -1,813,202 | -1,814,522 | -1,813,179 | 686 |
| hc5 | medium-graph | 12 | -1,605,897 | -1,607,260 | -1,614,102 | -1,588,028 | 8,090 |
| hc9 | medium-graph | 12 | -1,608,899 | -1,608,750 | -1,616,521 | -1,591,704 | 7,760 |
| hc11 | medium-graph | 12 | -1,607,611 | -1,607,706 | -1,615,244 | -1,590,960 | 7,747 |
| hc14 | medium-graph | 12 | -1,603,926 | -1,604,230 | -1,611,645 | -1,586,056 | 7,888 |
| hc5 | large-graph | 12 | -4,806,523 | -4,806,021 | -4,822,183 | -4,796,099 | 7,893 |
| hc9 | large-graph | 12 | -4,792,569 | -4,793,382 | -4,802,474 | -4,772,796 | 8,200 |
| hc11 | large-graph | 12 | -4,785,418 | -4,782,522 | -4,804,220 | -4,768,957 | 13,058 |
| hc14 | large-graph | 12 | -4,778,453 | -4,778,268 | -4,784,930 | -4,775,821 | 2,649 |

## 2. Pairwise test: hc9 vs each other variant, per problem

| problem | comparison | n_pairs | median Δ (hc9-other) | p-value |
|---|---|---:|---:|---:|
| small-graph | hc9 vs hc5 | 12 | -468 | 0.0034 |
| small-graph | hc9 vs hc11 | 12 | -561 | 0.0015 |
| small-graph | hc9 vs hc14 | 12 | -691 | 0.0005 |
| medium-graph | hc9 vs hc5 | 12 | -3,382 | 0.0005 |
| medium-graph | hc9 vs hc11 | 12 | -1,019 | 0.0005 |
| medium-graph | hc9 vs hc14 | 12 | -4,966 | 0.0005 |
| large-graph | hc9 vs hc5 | 12 | +15,096 | 0.0005 |
| large-graph | hc9 vs hc11 | 12 | -8,384 | 0.1763 |
| large-graph | hc9 vs hc14 | 12 | -15,996 | 0.0015 |

## 3. Suggested prose for the paper Discussion

> On small-graph the best mean over 12 seeds is hc9 (-1,814,574 ± 978 σ). On medium-graph the best mean over 12 seeds is hc9 (-1,608,899 ± 7,760 σ). On large-graph the best mean over 12 seeds is hc5 (-4,806,523 ± 7,893 σ).

Replace the discussion-paragraph 4 σ tuple with the actual values from §1.
