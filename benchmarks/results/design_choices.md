## Design choices with ML nuisances

100 seeds per dataset; default learners (regularised HGB).

| dataset                 | effect   | config                              |   truth |   mean_est |   bias |   rel_bias |   rmse |   coverage |   se_over_sd |   n_focal |    reps |
|:------------------------|:---------|:------------------------------------|--------:|-----------:|-------:|-----------:|-------:|-----------:|-------------:|----------:|--------:|
| flu_town (n=3000)       | direct   | default (IV-type, buffered, 2 hops) |  -2.000 |     -2.022 | -0.022 |     -0.011 |  0.170 |      0.950 |        1.027 |   471.380 | 100.000 |
| flu_town (n=3000)       | peer     | default (IV-type, buffered, 2 hops) |  -0.500 |     -0.496 |  0.004 |      0.009 |  0.087 |      0.930 |        1.040 |   471.380 | 100.000 |
| flu_town (n=3000)       | direct   | partialling-out score               |  -2.000 |     -1.962 |  0.038 |      0.019 |  0.184 |      0.920 |        0.977 |   471.380 | 100.000 |
| flu_town (n=3000)       | peer     | partialling-out score               |  -0.500 |     -0.470 |  0.030 |      0.060 |  0.089 |      0.970 |        1.104 |   471.380 | 100.000 |
| flu_town (n=3000)       | direct   | focal-only training                 |  -2.000 |     -2.011 | -0.011 |     -0.006 |  0.191 |      0.950 |        0.971 |   471.380 | 100.000 |
| flu_town (n=3000)       | peer     | focal-only training                 |  -0.500 |     -0.540 | -0.040 |     -0.080 |  0.153 |      0.950 |        0.872 |   471.380 | 100.000 |
| flu_town (n=3000)       | direct   | 1-hop features                      |  -2.000 |     -2.018 | -0.018 |     -0.009 |  0.171 |      0.970 |        1.008 |   471.380 | 100.000 |
| flu_town (n=3000)       | peer     | 1-hop features                      |  -0.500 |     -0.494 |  0.006 |      0.013 |  0.084 |      0.950 |        1.066 |   471.380 | 100.000 |
| flu_town (n=3000)       | direct   | partialling-out + 1-hop             |  -2.000 |     -1.965 |  0.035 |      0.017 |  0.186 |      0.930 |        0.950 |   471.380 | 100.000 |
| flu_town (n=3000)       | peer     | partialling-out + 1-hop             |  -0.500 |     -0.466 |  0.034 |      0.068 |  0.084 |      0.970 |        1.190 |   471.380 | 100.000 |
| flu_town (n=3000)       | direct   | original procedure                          |  -2.000 |     -1.947 |  0.053 |      0.027 |  0.257 |      0.910 |        0.895 |   398.370 | 100.000 |
| flu_town (n=3000)       | peer     | original procedure                          |  -0.500 |     -0.478 |  0.022 |      0.044 |  0.114 |      0.900 |        0.886 |   398.370 | 100.000 |
| referral_app (n=4000)   | direct   | default (IV-type, buffered, 2 hops) |  12.000 |     12.293 |  0.293 |      0.024 |  0.941 |      0.950 |        1.034 |   393.670 | 100.000 |
| referral_app (n=4000)   | peer     | default (IV-type, buffered, 2 hops) |  40.000 |     40.185 |  0.185 |      0.005 |  1.628 |      0.930 |        1.018 |   393.670 | 100.000 |
| referral_app (n=4000)   | direct   | partialling-out score               |  12.000 |     12.097 |  0.097 |      0.008 |  0.928 |      0.960 |        1.084 |   393.670 | 100.000 |
| referral_app (n=4000)   | peer     | partialling-out score               |  40.000 |     39.246 | -0.754 |     -0.019 |  1.936 |      0.930 |        0.975 |   393.670 | 100.000 |
| referral_app (n=4000)   | direct   | focal-only training                 |  12.000 |     13.203 |  1.203 |      0.100 |  1.553 |      0.800 |        1.044 |   393.670 | 100.000 |
| referral_app (n=4000)   | peer     | focal-only training                 |  40.000 |     40.694 |  0.694 |      0.017 |  2.507 |      0.890 |        0.832 |   393.670 | 100.000 |
| referral_app (n=4000)   | direct   | 1-hop features                      |  12.000 |     12.251 |  0.251 |      0.021 |  0.925 |      0.950 |        1.036 |   393.670 | 100.000 |
| referral_app (n=4000)   | peer     | 1-hop features                      |  40.000 |     40.231 |  0.231 |      0.006 |  1.668 |      0.910 |        0.992 |   393.670 | 100.000 |
| referral_app (n=4000)   | direct   | partialling-out + 1-hop             |  12.000 |     12.047 |  0.047 |      0.004 |  0.909 |      0.960 |        1.105 |   393.670 | 100.000 |
| referral_app (n=4000)   | peer     | partialling-out + 1-hop             |  40.000 |     39.340 | -0.660 |     -0.016 |  1.939 |      0.910 |        0.955 |   393.670 | 100.000 |
| referral_app (n=4000)   | direct   | original procedure                          |  12.000 |     13.656 |  1.656 |      0.138 |  2.161 |      0.750 |        0.925 |   344.900 | 100.000 |
| referral_app (n=4000)   | peer     | original procedure                          |  40.000 |     38.564 | -1.436 |     -0.036 |  3.334 |      0.840 |        0.777 |   344.900 | 100.000 |
| benchmark linear, ER n=3000 | direct   | default (IV-type, buffered, 2 hops) |  10.000 |      9.994 | -0.006 |     -0.001 |  0.084 |      0.960 |        1.005 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | peer     | default (IV-type, buffered, 2 hops) |   5.000 |      5.000 |  0.000 |      0.000 |  0.059 |      0.990 |        1.069 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | direct   | partialling-out score               |  10.000 |      9.977 | -0.023 |     -0.002 |  0.157 |      0.920 |        0.877 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | peer     | partialling-out score               |   5.000 |      4.974 | -0.026 |     -0.005 |  0.102 |      0.940 |        1.046 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | direct   | focal-only training                 |  10.000 |     10.002 |  0.002 |      0.000 |  0.087 |      0.960 |        1.039 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | peer     | focal-only training                 |   5.000 |      5.008 |  0.008 |      0.002 |  0.083 |      0.900 |        0.857 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | direct   | 1-hop features                      |  10.000 |      9.993 | -0.007 |     -0.001 |  0.081 |      0.940 |        1.028 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | peer     | 1-hop features                      |   5.000 |      4.999 | -0.001 |     -0.000 |  0.061 |      0.990 |        1.025 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | direct   | partialling-out + 1-hop             |  10.000 |      9.988 | -0.012 |     -0.001 |  0.140 |      0.940 |        0.948 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | peer     | partialling-out + 1-hop             |   5.000 |      4.973 | -0.027 |     -0.005 |  0.104 |      0.930 |        0.990 |   683.410 | 100.000 |
| benchmark linear, ER n=3000 | direct   | original procedure                          |  10.000 |      9.965 | -0.035 |     -0.004 |  0.370 |      0.950 |        0.981 |   548.460 | 100.000 |
| benchmark linear, ER n=3000 | peer     | original procedure                          |   5.000 |      4.825 | -0.175 |     -0.035 |  0.439 |      0.820 |        0.713 |   548.460 | 100.000 |
| benchmark nonlinear, ER n=3000 | direct   | default (IV-type, buffered, 2 hops) |  20.000 |     20.006 |  0.006 |      0.000 |  0.073 |      0.960 |        1.093 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | peer     | default (IV-type, buffered, 2 hops) |   5.000 |      5.003 |  0.003 |      0.001 |  0.050 |      0.960 |        1.168 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | direct   | partialling-out score               |  20.000 |     19.942 | -0.058 |     -0.003 |  0.218 |      0.910 |        0.850 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | peer     | partialling-out score               |   5.000 |      4.978 | -0.022 |     -0.004 |  0.110 |      1.000 |        1.234 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | direct   | focal-only training                 |  20.000 |     20.008 |  0.008 |      0.000 |  0.075 |      0.960 |        1.081 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | peer     | focal-only training                 |   5.000 |      5.011 |  0.011 |      0.002 |  0.061 |      0.950 |        1.010 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | direct   | 1-hop features                      |  20.000 |     20.004 |  0.004 |      0.000 |  0.073 |      0.950 |        1.086 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | peer     | 1-hop features                      |   5.000 |      5.003 |  0.003 |      0.001 |  0.051 |      0.970 |        1.149 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | direct   | partialling-out + 1-hop             |  20.000 |     19.971 | -0.029 |     -0.001 |  0.182 |      0.960 |        0.951 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | peer     | partialling-out + 1-hop             |   5.000 |      4.962 | -0.038 |     -0.008 |  0.126 |      0.950 |        1.057 |   683.410 | 100.000 |
| benchmark nonlinear, ER n=3000 | direct   | original procedure                          |  20.000 |     19.930 | -0.070 |     -0.004 |  0.387 |      0.920 |        0.984 |   548.460 | 100.000 |
| benchmark nonlinear, ER n=3000 | peer     | original procedure                          |   5.000 |      4.820 | -0.180 |     -0.036 |  0.427 |      0.840 |        0.756 |   548.460 | 100.000 |
