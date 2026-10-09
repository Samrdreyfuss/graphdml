# Signal Check (synthetic, planted effects)

**Question.** Does the estimator find a real effect when one exists, and stay quiet when
none does?

**Teaches.** What a trustworthy estimate looks like. The data come in two versions with the
same graph structure and the same strong network confounding:

| | Direct effect | Peer effect |
|---|---|---|
| **planted** (`make_signal_check()`) | +1.0 | +0.6 (share of treated neighbors, 0 to 100%) |
| **null twin** (`make_signal_check(direct_effect=0, peer_effect=0)`) | 0 | 0 |

Treatment and outcome both depend nonlinearly on a node's own covariates and on its
neighbors' (mean and maximum), so adjusting only for own covariates, or ignoring the
network, produces false signals. In the null twin, a method that ignores the network still
reports effects, because confounding looks like a treatment effect. A correct method should
report none.

Run the full check with `graphdml.selftest()`: it repeats both worlds many times and
reports bias, interval coverage, power and false-alarm rates against fixed criteria.

```python
import graphdml
print(graphdml.selftest())
```
