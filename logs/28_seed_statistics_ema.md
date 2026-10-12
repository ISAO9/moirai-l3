# 28 seed statistics (ema; F1-mean = mean of P and S F1 at ±20 ms)

Seed runs: 63; cells: 21; all cells have n = [3].
Training histories on disk by category: {'other': 2, 'array base': 8, 'array seed': 24, 'per-trace base': 8, 'per-trace seed': 24, 'shuffle base': 4, 'shuffle seed': 15} (total 85).
Regression gate against Section 4.2 (first revision): PASS

## Table A. All eight sites, array versus per-trace (three seeds; base run SEED 1234 for reference)

| Site | array F1-mean | per-trace F1-mean | array P / S | per-trace P / S | Δ per seed (array − per-trace) | Δ mean ± sd | same sign | bands overlap | max|Δ| < 0.02 | base array / per-trace |
|---|---|---|---|---|---|---|---|---|---|---|
| pnr-1 | 0.982 ± 0.005 | 0.985 ± 0.002 | 0.990 ± 0.003 / 0.975 ± 0.008 | 0.990 ± 0.001 / 0.980 ± 0.002 | +0.002 -0.003 -0.007 | -0.002 ± 0.005 | no | yes | yes | 0.984 / 0.988 |
| mseel_5h | 0.896 ± 0.003 | 0.885 ± 0.002 | 0.907 ± 0.006 / 0.885 ± 0.009 | 0.888 ± 0.002 / 0.881 ± 0.005 | +0.015 +0.009 +0.010 | +0.011 ± 0.003 | yes | no | yes | 0.895 / 0.885 |
| clearfield_mw6 | 0.876 ± 0.008 | 0.869 ± 0.004 | 0.878 ± 0.009 / 0.875 ± 0.008 | 0.868 ± 0.002 / 0.871 ± 0.006 | +0.010 +0.001 +0.008 | +0.007 ± 0.005 | yes | yes | yes | 0.873 / 0.870 |
| mseel_3h | 0.895 ± 0.004 | 0.890 ± 0.002 | 0.907 ± 0.005 / 0.883 ± 0.004 | 0.899 ± 0.000 / 0.882 ± 0.005 | +0.004 +0.002 +0.008 | +0.005 ± 0.003 | yes | yes | yes | 0.893 / 0.900 |
| pnr-2 | 0.874 ± 0.001 | 0.812 ± 0.011 | 0.873 ± 0.004 / 0.876 ± 0.003 | 0.875 ± 0.003 / 0.749 ± 0.025 | +0.067 +0.050 +0.070 | +0.063 ± 0.011 | yes | no | no | 0.881 / 0.840 |
| clearfield_mw4 | 0.858 ± 0.004 | 0.753 ± 0.018 | 0.870 ± 0.006 / 0.846 ± 0.002 | 0.776 ± 0.013 / 0.729 ± 0.025 | +0.086 +0.102 +0.129 | +0.106 ± 0.022 | yes | no | no | 0.852 / 0.774 |
| aneth | 0.885 ± 0.004 | 0.934 ± 0.012 | 0.954 ± 0.006 / 0.816 ± 0.010 | 0.956 ± 0.008 / 0.913 ± 0.016 | -0.059 -0.034 -0.055 | -0.049 ± 0.013 | yes | no | no | 0.889 / 0.940 |
| forge_19 | 0.458 ± 0.107 | 0.807 ± 0.050 | 0.121 ± 0.088 / 0.796 ± 0.134 | 0.696 ± 0.100 / 0.919 ± 0.014 | -0.466 -0.259 -0.323 | -0.349 ± 0.106 | yes | no | no | 0.518 / 0.842 |

## Table B. Station-shuffle control (mean ± sd over three seeds) -- manuscript Table 4; the mseel_5h row appears only if its shuffle runs exist

| Site | array P / S / mean | shuffle P / S / mean | per-trace P / S / mean |
|---|---|---|---|
| forge_19 | 0.121 ± 0.088 / 0.796 ± 0.134 / 0.458 ± 0.107 | 0.378 ± 0.160 / 0.750 ± 0.126 / 0.564 ± 0.129 | 0.696 ± 0.100 / 0.919 ± 0.014 / 0.807 ± 0.050 |
| clearfield_mw4 | 0.870 ± 0.006 / 0.846 ± 0.002 / 0.858 ± 0.004 | 0.863 ± 0.004 / 0.838 ± 0.008 / 0.851 ± 0.005 | 0.776 ± 0.013 / 0.729 ± 0.025 / 0.753 ± 0.018 |
| pnr-2 | 0.873 ± 0.004 / 0.876 ± 0.003 / 0.874 ± 0.001 | 0.887 ± 0.005 / 0.788 ± 0.025 / 0.838 ± 0.010 | 0.875 ± 0.003 / 0.749 ± 0.025 / 0.812 ± 0.011 |
| aneth | 0.954 ± 0.006 / 0.816 ± 0.010 / 0.885 ± 0.004 | 0.953 ± 0.002 / 0.912 ± 0.011 / 0.933 ± 0.007 | 0.956 ± 0.008 / 0.913 ± 0.016 / 0.934 ± 0.012 |
| mseel_5h | 0.907 ± 0.006 / 0.885 ± 0.009 / 0.896 ± 0.003 | 0.905 ± 0.009 / 0.881 ± 0.008 / 0.893 ± 0.009 | 0.888 ± 0.002 / 0.881 ± 0.005 / 0.885 ± 0.002 |

## Table C. Paired-by-seed differences at the shuffle sites (F1-mean)

| Site | pair | Δ per seed | Δ mean ± sd | same sign | bands overlap |
|---|---|---|---|---|---|
| forge_19 | array − shuffle | -0.309 +0.086 -0.093 | -0.105 ± 0.198 | no | yes |
| forge_19 | per-trace − shuffle | +0.157 +0.345 +0.230 | +0.244 ± 0.095 | yes | no |
| forge_19 | array − per-trace | -0.466 -0.259 -0.323 | -0.349 ± 0.106 | yes | no |
| clearfield_mw4 | array − shuffle | +0.003 +0.014 +0.005 | +0.007 ± 0.006 | yes | yes |
| clearfield_mw4 | per-trace − shuffle | -0.083 -0.088 -0.124 | -0.098 ± 0.022 | yes | no |
| clearfield_mw4 | array − per-trace | +0.086 +0.102 +0.129 | +0.106 ± 0.022 | yes | no |
| pnr-2 | array − shuffle | +0.032 +0.049 +0.029 | +0.037 ± 0.011 | yes | no |
| pnr-2 | per-trace − shuffle | -0.035 -0.001 -0.041 | -0.026 ± 0.021 | yes | no |
| pnr-2 | array − per-trace | +0.067 +0.050 +0.070 | +0.063 ± 0.011 | yes | no |
| aneth | array − shuffle | -0.052 -0.046 -0.045 | -0.048 ± 0.004 | yes | no |
| aneth | per-trace − shuffle | +0.006 -0.012 +0.010 | +0.002 ± 0.012 | no | yes |
| aneth | array − per-trace | -0.059 -0.034 -0.055 | -0.049 ± 0.013 | yes | no |
| mseel_5h | array − shuffle | +0.007 +0.010 -0.010 | +0.003 ± 0.011 | no | yes |
| mseel_5h | per-trace − shuffle | -0.008 +0.002 -0.019 | -0.009 ± 0.010 | no | yes |
| mseel_5h | array − per-trace | +0.015 +0.009 +0.010 | +0.011 ± 0.003 | yes | no |

## Table D. Per-phase paired differences at the shuffle sites (array − shuffle)

| Site | ΔP per seed (mean) | ΔS per seed (mean) |
|---|---|---|
| forge_19 | -0.498 -0.090 -0.181 (-0.256) | -0.120 +0.262 -0.005 (+0.046) |
| clearfield_mw4 | -0.002 +0.012 +0.010 (+0.007) | +0.008 +0.016 -0.001 (+0.008) |
| pnr-2 | -0.013 -0.016 -0.014 (-0.014) | +0.077 +0.114 +0.073 (+0.088) |
| aneth | -0.007 +0.005 +0.005 (+0.001) | -0.097 -0.096 -0.095 (-0.096) |
| mseel_5h | -0.001 +0.016 -0.009 (+0.002) | +0.014 +0.005 -0.010 (+0.003) |

## Sentences for the manuscript (three decimals, seed runs only)

- pnr-1: array 0.982 ± 0.005 versus per-trace 0.985 ± 0.002
- mseel_5h: array 0.896 ± 0.003 versus per-trace 0.885 ± 0.002
- clearfield_mw6: array 0.876 ± 0.008 versus per-trace 0.869 ± 0.004
- mseel_3h: array 0.895 ± 0.004 versus per-trace 0.890 ± 0.002
- pnr-2: array 0.874 ± 0.001 versus per-trace 0.812 ± 0.011
- clearfield_mw4: array 0.858 ± 0.004 versus per-trace 0.753 ± 0.018
- aneth: array 0.885 ± 0.004 versus per-trace 0.934 ± 0.012
- forge_19: array 0.458 ± 0.107 versus per-trace 0.807 ± 0.050
- forge_19 (shuffle): array 0.458 ± 0.107 → shuffle 0.564 ± 0.129 → per-trace 0.807 ± 0.050; P 0.121 ± 0.088 → 0.378 ± 0.160 → 0.696 ± 0.100; S 0.796 ± 0.134 → 0.750 ± 0.126 → 0.919 ± 0.014
- clearfield_mw4 (shuffle): array 0.858 ± 0.004 → shuffle 0.851 ± 0.005 → per-trace 0.753 ± 0.018; P 0.870 ± 0.006 → 0.863 ± 0.004 → 0.776 ± 0.013; S 0.846 ± 0.002 → 0.838 ± 0.008 → 0.729 ± 0.025
- pnr-2 (shuffle): array 0.874 ± 0.001 → shuffle 0.838 ± 0.010 → per-trace 0.812 ± 0.011; P 0.873 ± 0.004 → 0.887 ± 0.005 → 0.875 ± 0.003; S 0.876 ± 0.003 → 0.788 ± 0.025 → 0.749 ± 0.025
- aneth (shuffle): array 0.885 ± 0.004 → shuffle 0.933 ± 0.007 → per-trace 0.934 ± 0.012; P 0.954 ± 0.006 → 0.953 ± 0.002 → 0.956 ± 0.008; S 0.816 ± 0.010 → 0.912 ± 0.011 → 0.913 ± 0.016
- mseel_5h (shuffle): array 0.896 ± 0.003 → shuffle 0.893 ± 0.009 → per-trace 0.885 ± 0.002; P 0.907 ± 0.006 → 0.905 ± 0.009 → 0.888 ± 0.002; S 0.885 ± 0.009 → 0.881 ± 0.008 → 0.881 ± 0.005
