# LastFM Promo (semi-synthetic, real network)

**Question.** Does a concert promotion increase a user's listening hours, and does it spill
over to friends who did not get it?

**Teaches.** How GraphDML behaves on a *real* social network: real topology, real covariates,
and a heavily skewed degree distribution. Only the treatment and outcome are simulated, so
the truth is known.

| | |
|---|---|
| Nodes | 7,624 LastFM users from Asian countries (real) |
| Edges | 27,806 mutual-follow relationships (real; mean degree 7.3, max 216) |
| Covariates | `activity` (log number of liked artists), `taste_1..6` (SVD of liked artists) (real) |
| Treatment | `promo` (0/1, simulated): targeted on own and friends' taste and activity |
| Outcome | `listening_hours` per month (simulated) |
| Exposure map | `"mean"`: share of friends who received the promo |
| True direct effect | **+2.0** hours |
| True peer effect | **+3.0** hours going from 0% to 100% of friends promoted |

**What confounds.** The targeting rule favours users whose friends share a particular taste,
and friends' taste also drives listening (nonlinearly, through the friends' maximum of one
taste dimension).

**What to notice.** The focal set (about 21% of users) is dominated by users with one or two
friends: mean degree 1.3 versus 7.3 overall. With constant effects that is harmless. With
effects that vary by popularity, it would change what is being estimated.

```python
from graphdml import GraphDML
from graphdml.datasets import make_lastfm_promo

ds = make_lastfm_promo()          # downloads ~6.5 MB from SNAP on first use
print(GraphDML(exposure="mean", random_state=0).fit(ds.data).summary())
print(ds.extras["citation"])
```

Data: Rozemberczki & Sarkar (2020), via SNAP. graphdml downloads it on demand and does not
redistribute it.
