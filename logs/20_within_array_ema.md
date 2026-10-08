| Site | test events | LOSO array / per-trace | within-array array / per-trace | array - per-trace, paired 95% CI | LOSO verdict | within-array verdict |
|---|---|---|---|---|---|---|
| pnr-1 | 1258 | 0.986 / 0.986 | 0.985 / 0.986 | -0.001 [-0.005, +0.002] | tie | tie |
| mseel_3h | 253 | 0.891 / 0.888 | 0.851 / 0.846 | +0.005 [-0.004, +0.016] | tie | tie |
| mseel_5h | 221 | 0.897 / 0.881 | 0.948 / 0.941 | +0.006 [+0.001, +0.014] | tie | array |
| clearfield_mw6 | 228 | 0.875 / 0.869 | 0.871 / 0.890 | -0.017 [-0.036, -0.003] | tie | per-trace |
| pnr-2 | 159 | 0.944 / 0.901 | 0.975 / 0.965 | +0.010 [+0.002, +0.020] | array | array |
| clearfield_mw4 | 171 | 0.852 / 0.836 | 0.903 / 0.909 | -0.006 [-0.019, +0.005] | tie | tie |
| aneth | 47 | 0.884 / 0.941 | 0.974 / 0.971 | +0.003 [-0.003, +0.009] | per-trace | tie |
| forge_19 | 193 | 0.516 / 0.856 | 0.950 / 0.952 | -0.002 [-0.011, +0.008] | per-trace | tie |

At forge_19 the per-trace advantage is +0.340 under leave-one-site-out and +0.002 when the site's own events are in the training split.
The gap closes once the moveout is represented in training, which is what the paper's mechanism predicts: the collapse is a property of the training distribution, not of the site.