# Synthetic-benchmark analysis (multi-seed)

- Instances: 20
- Variants: 13 (hc1, hc2, hc3, hc4, hc5, hc6, hc7, hc8, hc9, hc10, hc11, hc12, hc13)
- Seeds per (variant, instance): [3]
- Statistics package: stdlib only (Friedman χ² + p-value via pure-Python incomplete-gamma χ² survival function)

## 1. Mean rank across instances
(rank 1 = best on that instance; lower mean rank = better overall)

| variant | mean rank | wins by mean |
|---|---:|---:|
| hc9 | 2.35 | 10 |
| hc11 | 4.00 | 0 |
| hc12 | 4.05 | 6 |
| hc13 | 4.25 | 0 |
| hc7 | 4.75 | 0 |
| hc5 | 5.90 | 0 |
| hc4 | 6.30 | 0 |
| hc8 | 7.60 | 0 |
| hc10 | 7.80 | 1 |
| hc6 | 8.75 | 0 |
| hc2 | 10.75 | 3 |
| hc3 | 12.00 | 0 |
| hc1 | 12.50 | 0 |

## 2. Friedman test (mean ranks differ across variants?)

- χ² = 167.505, df = 12, p = 1.5447e-29

## 3. Nemenyi critical difference
- CD(α=0.05, k=13, N=20) = 4.080
- Variants whose mean ranks differ by more than CD are significantly different (Demšar 2006).

## 4. Per-instance ranks (compact)

| instance | hc1 | hc2 | hc3 | hc4 | hc5 | hc6 | hc7 | hc8 | hc9 | hc10 | hc11 | hc12 | hc13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inst_01_n200_d3 | 13.0 | 11.0 | 12.0 | 5.0 | 6.0 | 7.0 | 4.0 | 9.0 | 1.0 | 8.0 | 2.0 | 10.0 | 3.0 |
| inst_02_n200_d8 | 13.0 | 12.0 | 11.0 | 7.0 | 4.0 | 9.0 | 3.0 | 8.0 | 1.0 | 10.0 | 2.0 | 6.0 | 5.0 |
| inst_03_n200_d18 | 13.0 | 12.0 | 11.0 | 6.0 | 7.0 | 9.0 | 5.0 | 8.0 | 1.0 | 10.0 | 4.0 | 2.0 | 3.0 |
| inst_04_n200_d35 | 12.0 | 13.0 | 11.0 | 7.0 | 6.0 | 10.0 | 5.0 | 8.0 | 2.0 | 9.0 | 3.0 | 1.0 | 4.0 |
| inst_05_n350_d3 | 13.0 | 11.0 | 12.0 | 6.0 | 3.0 | 10.0 | 5.0 | 8.0 | 1.0 | 7.0 | 2.0 | 9.0 | 4.0 |
| inst_06_n350_d8 | 13.0 | 12.0 | 11.0 | 6.0 | 3.0 | 9.0 | 7.0 | 8.0 | 1.0 | 10.0 | 5.0 | 2.0 | 4.0 |
| inst_07_n350_d18 | 13.0 | 12.0 | 11.0 | 8.0 | 6.0 | 9.0 | 5.0 | 7.0 | 1.0 | 10.0 | 4.0 | 2.0 | 3.0 |
| inst_08_n350_d35 | 13.0 | 12.0 | 11.0 | 8.0 | 7.0 | 9.0 | 5.0 | 6.0 | 1.0 | 10.0 | 4.0 | 2.0 | 3.0 |
| inst_09_n500_d3 | 12.0 | 11.0 | 13.0 | 7.0 | 6.0 | 10.0 | 2.0 | 8.0 | 3.0 | 1.0 | 5.0 | 9.0 | 4.0 |
| inst_10_n500_d8 | 12.0 | 11.0 | 13.0 | 6.0 | 7.0 | 9.0 | 4.0 | 8.0 | 1.0 | 10.0 | 3.0 | 2.0 | 5.0 |
| inst_11_n500_d18 | 13.0 | 12.0 | 11.0 | 7.0 | 3.0 | 9.0 | 5.0 | 8.0 | 2.0 | 10.0 | 4.0 | 1.0 | 6.0 |
| inst_12_n500_d35 | 12.0 | 13.0 | 11.0 | 6.0 | 4.0 | 10.0 | 3.0 | 8.0 | 2.0 | 9.0 | 5.0 | 1.0 | 7.0 |
| inst_13_n750_d3 | 13.0 | 11.0 | 12.0 | 7.0 | 8.0 | 10.0 | 3.0 | 9.0 | 1.0 | 2.0 | 5.0 | 6.0 | 4.0 |
| inst_14_n750_d8 | 13.0 | 11.0 | 12.0 | 5.0 | 8.0 | 9.0 | 6.0 | 7.0 | 2.0 | 10.0 | 3.5 | 1.0 | 3.5 |
| inst_15_n750_d18 | 12.0 | 11.0 | 13.0 | 5.0 | 8.0 | 9.0 | 4.0 | 7.0 | 6.0 | 10.0 | 3.0 | 1.0 | 2.0 |
| inst_16_n750_d35 | 12.0 | 6.0 | 13.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| inst_17_n1000_d3 | 12.0 | 11.0 | 13.0 | 5.0 | 8.0 | 10.0 | 6.0 | 9.0 | 1.0 | 2.0 | 4.0 | 7.0 | 3.0 |
| inst_18_n1000_d8 | 12.0 | 11.0 | 13.0 | 7.0 | 6.0 | 9.0 | 5.0 | 8.0 | 2.0 | 10.0 | 3.5 | 1.0 | 3.5 |
| inst_19_n1000_d18 | 12.0 | 6.0 | 13.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| inst_20_n1000_d35 | 12.0 | 6.0 | 13.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 | 6.0 |

## 5. Suggested prose for the paper Discussion

> Across 20 synthetic Erdős–Rényi instances and 3 seeds per cell, the variants with the lowest mean rank are hc9 (2.35), hc11 (4.00), hc12 (4.05). Wins by per-cell mean (most-negative average score): hc9=10, hc12=6, hc10=1, hc2=3.
