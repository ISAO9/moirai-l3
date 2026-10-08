| Site | events | array F1-mean (submitted -> paired) | per-trace F1-mean (submitted -> paired) | array - per-trace, paired 95% CI | verdict, submitted | verdict, marginal on paired data | verdict, paired | direction reversed |
|---|---|---|---|---|---|---|---|---|
| pnr-1 | 1258 | 0.986 -> 0.986 | 0.986 -> 0.985 | +0.002 [-0.001, +0.004] | tie | tie | tie | no |
| mseel_3h | 1684 | 0.891 -> 0.900 | 0.888 -> 0.901 | -0.001 [-0.007, +0.003] | tie | tie | tie | no |
| mseel_5h | 1512 | 0.897 -> 0.889 | 0.881 -> 0.880 | +0.009 [+0.001, +0.017] | tie | tie | array | no |
| clearfield_mw6 | 1612 | 0.875 -> 0.870 | 0.869 -> 0.868 | +0.002 [-0.005, +0.008] | tie | tie | tie | no |
| pnr-2 | 972 | 0.944 -> 0.940 | 0.901 -> 0.904 | +0.036 [+0.027, +0.046] | array | array | array | no |
| clearfield_mw4 | 1254 | 0.852 -> 0.858 | 0.836 -> 0.839 | +0.020 [+0.009, +0.030] | tie | tie | array | no |
| aneth | 298 | 0.884 -> 0.887 | 0.941 -> 0.942 | -0.055 [-0.065, -0.047] | per-trace | per-trace | per-trace | no |
| forge_19 | 1213 | 0.516 -> 0.524 | 0.856 -> 0.856 | -0.332 [-0.355, -0.310] | per-trace | per-trace | per-trace | no |

Evaluating both configurations on identical windows and identical sensors moves no site-level F1-mean by more than 0.013, so the independent draws used in the submitted evaluation cost essentially nothing.

Against the submitted verdicts: 0 of 8 reverse direction, 6 are identical, and 2 move from a tie to a decided verdict. A tie becoming decided is the expected effect of pairing, which cancels the between-event variance the two configurations share and is therefore the more powerful test; it is not a disagreement with the submitted result. The marginal column is recomputed on the SAME paired caches and is shown only so the two interval forms can be compared on one dataset.