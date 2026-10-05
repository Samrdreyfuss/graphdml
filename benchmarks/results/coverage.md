## Coverage gate

Stage 1 (screen): 300 seeds, n = 2000 nodes, nominal 95% intervals. Band 0.95 ± 0.038 (3 Monte Carlo SEs). Stage 2 reran 1 flagged cell(s): see coverage_confirm.md.

| dgp                    | nuisances        | graph           | effect   |   truth |   mean_est |   bias |   rel_bias |   rmse |   coverage |   se_over_sd |   n_focal |    reps | in_band   |
|:-----------------------|:-----------------|:----------------|:---------|--------:|-----------:|-------:|-----------:|-------:|-----------:|-------------:|----------:|--------:|:----------|
| linear, continuous T   | oracle           | ER (avg deg 4)  | direct   |   1.000 |      1.002 |  0.002 |      0.002 |  0.052 |      0.920 |        0.904 |   454.237 | 300.000 | True      |
| linear, continuous T   | oracle           | ER (avg deg 4)  | peer     |   0.500 |      0.500 |  0.000 |      0.000 |  0.061 |      0.940 |        1.016 |   454.237 | 300.000 | True      |
| linear, continuous T   | linear (correct) | ER (avg deg 4)  | direct   |   1.000 |      1.002 |  0.002 |      0.002 |  0.052 |      0.923 |        0.915 |   454.237 | 300.000 | True      |
| linear, continuous T   | linear (correct) | ER (avg deg 4)  | peer     |   0.500 |      0.501 |  0.001 |      0.002 |  0.061 |      0.930 |        1.012 |   454.237 | 300.000 | True      |
| paper eq. 38, binary T | oracle           | ER (avg deg 4)  | direct   |  10.000 |     10.001 |  0.001 |      0.000 |  0.085 |      0.970 |        1.107 |   454.237 | 300.000 | True      |
| paper eq. 38, binary T | oracle           | ER (avg deg 4)  | peer     |   5.000 |      5.003 |  0.003 |      0.001 |  0.076 |      0.950 |        0.929 |   454.237 | 300.000 | True      |
| linear, continuous T   | oracle           | BA (m=2, hubs)  | direct   |   1.000 |      1.005 |  0.005 |      0.005 |  0.060 |      0.937 |        0.946 |   310.237 | 300.000 | True      |
| linear, continuous T   | oracle           | BA (m=2, hubs)  | peer     |   0.500 |      0.507 |  0.007 |      0.013 |  0.090 |      0.927 |        0.913 |   310.237 | 300.000 | True      |
| linear, continuous T   | linear (correct) | BA (m=2, hubs)  | direct   |   1.000 |      1.005 |  0.005 |      0.005 |  0.061 |      0.940 |        0.933 |   310.237 | 300.000 | True      |
| linear, continuous T   | linear (correct) | BA (m=2, hubs)  | peer     |   0.500 |      0.507 |  0.007 |      0.014 |  0.091 |      0.907 |        0.901 |   310.237 | 300.000 | False     |
| paper eq. 38, binary T | oracle           | BA (m=2, hubs)  | direct   |  10.000 |     10.005 |  0.005 |      0.000 |  0.116 |      0.950 |        0.993 |   310.237 | 300.000 | True      |
| paper eq. 38, binary T | oracle           | BA (m=2, hubs)  | peer     |   5.000 |      5.001 |  0.001 |      0.000 |  0.078 |      0.960 |        1.031 |   310.237 | 300.000 | True      |
| linear, continuous T   | oracle           | SBM (40 blocks) | direct   |   1.000 |      1.000 |  0.000 |      0.000 |  0.068 |      0.947 |        0.965 |   231.577 | 300.000 | True      |
| linear, continuous T   | oracle           | SBM (40 blocks) | peer     |   0.500 |      0.491 | -0.009 |     -0.018 |  0.109 |      0.970 |        1.044 |   231.577 | 300.000 | True      |
| linear, continuous T   | linear (correct) | SBM (40 blocks) | direct   |   1.000 |      1.000 |  0.000 |      0.000 |  0.068 |      0.953 |        0.969 |   231.577 | 300.000 | True      |
| linear, continuous T   | linear (correct) | SBM (40 blocks) | peer     |   0.500 |      0.491 | -0.009 |     -0.019 |  0.108 |      0.957 |        1.053 |   231.577 | 300.000 | True      |
| paper eq. 38, binary T | oracle           | SBM (40 blocks) | direct   |  10.000 |     10.005 |  0.005 |      0.001 |  0.142 |      0.940 |        0.954 |   231.577 | 300.000 | True      |
| paper eq. 38, binary T | oracle           | SBM (40 blocks) | peer     |   5.000 |      5.001 |  0.001 |      0.000 |  0.075 |      0.947 |        0.981 |   231.577 | 300.000 | True      |
