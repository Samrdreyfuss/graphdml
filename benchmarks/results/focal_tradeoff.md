## The cost of the focal set

100 seeds; default learners.

| dataset               | effect   | method                                   |   truth |   mean_est |   bias |   rel_bias |   rmse |   coverage |   se_over_sd |   n_focal |    reps |
|:----------------------|:---------|:-----------------------------------------|--------:|-----------:|-------:|-----------:|-------:|-----------:|-------------:|----------:|--------:|
| flu_town (n=3000)     | direct   | GraphDML (focal set)                     |  -2.000 |     -2.022 | -0.022 |     -0.011 |  0.170 |      0.950 |        1.027 |   471.380 | 100.000 |
| flu_town (n=3000)     | peer     | GraphDML (focal set)                     |  -0.500 |     -0.496 |  0.004 |      0.009 |  0.087 |      0.930 |        1.040 |   471.380 | 100.000 |
| flu_town (n=3000)     | direct   | GraphDML features, all nodes             |  -2.000 |     -2.016 | -0.016 |     -0.008 |  0.062 |      0.980 |        1.112 |  3000.000 | 100.000 |
| flu_town (n=3000)     | peer     | GraphDML features, all nodes             |  -0.500 |     -0.498 |  0.002 |      0.005 |  0.031 |      0.880 |        0.866 |  3000.000 | 100.000 |
| flu_town (n=3000)     | direct   | 1-hop aggregates, all nodes (paper's PA) |  -2.000 |     -2.015 | -0.015 |     -0.007 |  0.063 |      0.980 |        1.084 |  3000.000 | 100.000 |
| flu_town (n=3000)     | peer     | 1-hop aggregates, all nodes (paper's PA) |  -0.500 |     -0.498 |  0.002 |      0.005 |  0.031 |      0.880 |        0.836 |  3000.000 | 100.000 |
| referral_app (n=4000) | direct   | GraphDML (focal set)                     |  12.000 |     12.293 |  0.293 |      0.024 |  0.941 |      0.950 |        1.034 |   393.670 | 100.000 |
| referral_app (n=4000) | peer     | GraphDML (focal set)                     |  40.000 |     40.185 |  0.185 |      0.005 |  1.628 |      0.930 |        1.018 |   393.670 | 100.000 |
| referral_app (n=4000) | direct   | GraphDML features, all nodes             |  12.000 |     12.171 |  0.171 |      0.014 |  0.364 |      0.900 |        0.893 |  4000.000 | 100.000 |
| referral_app (n=4000) | peer     | GraphDML features, all nodes             |  40.000 |     40.163 |  0.163 |      0.004 |  0.656 |      0.920 |        0.928 |  4000.000 | 100.000 |
| referral_app (n=4000) | direct   | 1-hop aggregates, all nodes (paper's PA) |  12.000 |     12.138 |  0.138 |      0.012 |  0.343 |      0.910 |        0.913 |  4000.000 | 100.000 |
| referral_app (n=4000) | peer     | 1-hop aggregates, all nodes (paper's PA) |  40.000 |     40.132 |  0.132 |      0.003 |  0.643 |      0.920 |        0.933 |  4000.000 | 100.000 |
