# Paired inference on the array-minus-per-trace difference

Delta = F1-mean(array) - F1-mean(per-trace). B = 2000 joint event-bootstrap resamples; one resampled index set is scored under both configurations, so Delta is paired within each resample. Equivalence bound ±0.02 on F1-mean.

## false alarms excluded

| Site | array | per-trace | Delta | 95 % CI on Delta | Pr(Delta>0) | Paired verdict |
|---|---|---|---|---|---|---|
| aneth | 0.884 | 0.941 | -0.057 | [-0.069, -0.047] | 0.000 | different |
| clearfield_mw4 | 0.852 | 0.836 | +0.016 | [-0.009, +0.040] | 0.889 | tie (inconclusive) |
| clearfield_mw6 | 0.875 | 0.869 | +0.006 | [-0.006, +0.018] | 0.822 | equivalent |
| forge_19 | 0.516 | 0.856 | -0.340 | [-0.362, -0.317] | 0.000 | different |
| mseel_3h | 0.891 | 0.888 | +0.003 | [-0.009, +0.017] | 0.711 | equivalent |
| mseel_5h | 0.897 | 0.881 | +0.016 | [+0.004, +0.029] | 0.997 | different |
| pnr-1 | 0.986 | 0.986 | +0.000 | [-0.005, +0.006] | 0.539 | equivalent |
| pnr-2 | 0.944 | 0.901 | +0.043 | [+0.024, +0.063] | 1.000 | different |

Across the eight sites: median Delta +0.005, site-mean -0.039; exact sign test p = 0.289 (6 of 8 positive); Wilcoxon signed-rank p = 0.742.

## false alarms included

| Site | array | per-trace | Delta | 95 % CI on Delta | Pr(Delta>0) | Paired verdict |
|---|---|---|---|---|---|---|
| aneth | 0.884 | 0.941 | -0.057 | [-0.069, -0.047] | 0.000 | different |
| clearfield_mw4 | 0.852 | 0.773 | +0.078 | [+0.053, +0.105] | 1.000 | different |
| clearfield_mw6 | 0.875 | 0.866 | +0.009 | [-0.003, +0.021] | 0.931 | tie (inconclusive) |
| forge_19 | 0.514 | 0.840 | -0.325 | [-0.348, -0.302] | 0.000 | different |
| mseel_3h | 0.891 | 0.887 | +0.005 | [-0.008, +0.018] | 0.764 | equivalent |
| mseel_5h | 0.897 | 0.879 | +0.017 | [+0.005, +0.030] | 0.999 | different |
| pnr-1 | 0.985 | 0.985 | -0.001 | [-0.006, +0.005] | 0.383 | equivalent |
| pnr-2 | 0.886 | 0.831 | +0.055 | [+0.033, +0.076] | 1.000 | different |

Across the eight sites: median Delta +0.007, site-mean -0.027; exact sign test p = 0.727 (5 of 8 positive); Wilcoxon signed-rank p = 0.742.
