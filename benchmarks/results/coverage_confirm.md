## Coverage gate, stage 2 (confirmation)

Cells outside the stage-1 band, rerun with 2000 fresh seeds. Band 0.95 ± 0.015.

| dgp                  | nuisances        | graph          | effect   |   truth |   mean_est |   bias |   rel_bias |   rmse |   coverage |   se_over_sd |   n_focal |     reps | in_band   |
|:---------------------|:-----------------|:---------------|:---------|--------:|-----------:|-------:|-----------:|-------:|-----------:|-------------:|----------:|---------:|:----------|
| linear, continuous T | linear (correct) | BA (m=2, hubs) | direct   |   1.000 |      1.001 |  0.001 |      0.001 |  0.058 |      0.939 |        0.994 |   310.327 | 2000.000 | True      |
| linear, continuous T | linear (correct) | BA (m=2, hubs) | peer     |   0.500 |      0.500 | -0.000 |     -0.001 |  0.083 |      0.944 |        0.982 |   310.327 | 2000.000 | True      |
