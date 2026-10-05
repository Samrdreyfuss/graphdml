## Semi-synthetic experiment on the LastFM Asia network

100 replications; real graph and covariates, simulated promo and listening hours; truth: direct 2.0, peer 3.0 (exposure: share of friends).

| effect   | method                                 |   truth |   mean_est |   bias |   rel_bias |   rmse |   coverage |   se_over_sd |   n_focal |    reps |
|:---------|:---------------------------------------|--------:|-----------:|-------:|-----------:|-------:|-----------:|-------------:|----------:|--------:|
| direct   | GraphDML (default)                     |   2.000 |      2.045 |  0.045 |      0.023 |  0.192 |      0.920 |        0.996 |  1613.480 | 100.000 |
| peer     | GraphDML (default)                     |   3.000 |      3.019 |  0.019 |      0.006 |  0.194 |      0.960 |        0.987 |  1613.480 | 100.000 |
| direct   | GraphDML (random-order focal set)      |   2.000 |      2.037 |  0.037 |      0.019 |  0.222 |      0.930 |        0.948 |  1289.330 | 100.000 |
| peer     | GraphDML (random-order focal set)      |   3.000 |      2.997 | -0.003 |     -0.001 |  0.218 |      0.970 |        1.064 |  1289.330 | 100.000 |
| direct   | GraphDML (original procedure)                  |   2.000 |      2.124 |  0.124 |      0.062 |  0.272 |      0.860 |        0.922 |  1289.330 | 100.000 |
| peer     | GraphDML (original procedure)                  |   3.000 |      2.810 | -0.190 |     -0.063 |  0.300 |      0.880 |        1.061 |  1289.330 | 100.000 |
| direct   | DML + 1-hop aggregates, all nodes |   2.000 |      2.021 |  0.021 |      0.010 |  0.090 |      0.930 |        0.941 |  7624.000 | 100.000 |
| peer     | DML + 1-hop aggregates, all nodes |   3.000 |      3.007 |  0.007 |      0.002 |  0.122 |      0.950 |        1.026 |  7624.000 | 100.000 |
| direct   | DML, own covariates (i.i.d.)           |   2.000 |      3.113 |  1.113 |      0.557 |  1.118 |      0.000 |        0.995 |  7624.000 | 100.000 |
| direct   | OLS + exposure (no network)            |   2.000 |      2.742 |  0.742 |      0.371 |  0.749 |      0.000 |        0.938 |   nan     | 100.000 |
| peer     | OLS + exposure (no network)            |   3.000 |      5.138 |  2.138 |      0.713 |  2.144 |      0.000 |        0.902 |   nan     | 100.000 |
