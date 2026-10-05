# Benchmarks

Monte Carlo evidence for [the validation protocol](../docs/validation.md). Each script runs seeds in
parallel (one thread per worker), writes per-seed rows to `results/<name>.csv` and a
summary to `results/<name>.md`. `report.py` assembles `docs/validation-results.md`.

| Script | Question | Typical runtime (10 cores) |
|---|---|---|
| `coverage.py` | Do 95% intervals cover at 95% with oracle and correctly specified nuisances? Two-stage gate. | ~2 min |
| `design_choices.py` | Which defaults (score, nuisance training, feature hops) work best with ML nuisances? | ~5 min |
| `lastfm_experiment.py` | Does GraphDML recover known effects on a real social network (semi-synthetic)? | ~3 min |
| `focal_tradeoff.py` | What does restricting to the focal set cost in efficiency, and what does ignoring dependence cost in coverage? | ~3 min |

```bash
pip install -e ".[dev]"
cd benchmarks
python coverage.py && python design_choices.py && python focal_tradeoff.py \
  && python lastfm_experiment.py && python report.py
```
