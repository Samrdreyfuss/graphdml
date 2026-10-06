## Total effect (ATE) coverage

Truth = direct + peer × average exposure when all nodes are treated.

| setting                           |   reps |   truth |   mean_est |   rel_bias |   coverage |   se_over_sd | band         |
|:----------------------------------|-------:|--------:|-----------:|-----------:|-----------:|-------------:|:-------------|
| ER, mean, oracle                  |    300 |   1.491 |      1.493 |      0.001 |      0.953 |        0.980 | 0.95 ± 0.038 |
| ER, mean, linear (correct)        |    300 |   1.491 |      1.494 |      0.002 |      0.950 |        0.996 | 0.95 ± 0.038 |
| ER, sum, oracle                   |    300 |   2.995 |      2.994 |     -0.000 |      0.957 |        0.978 | 0.95 ± 0.038 |
| BA (hubs), mean, oracle           |    300 |   1.500 |      1.512 |      0.008 |      0.920 |        0.913 | 0.95 ± 0.038 |
| BA (hubs), mean, linear (correct) |    300 |   1.500 |      1.512 |      0.008 |      0.913 |        0.900 | 0.95 ± 0.038 |
| BA (hubs), sum, oracle            |    300 |   2.998 |      3.016 |      0.006 |      0.923 |        0.906 | 0.95 ± 0.038 |
| flu_town (n=3000), HGB            |    100 |  -4.933 |     -4.930 |      0.001 |      0.940 |        0.988 | 0.95 ± 0.065 |
| referral_app (n=4000), HGB        |    100 |  52.000 |     52.478 |      0.009 |      0.900 |        0.941 | 0.95 ± 0.065 |
