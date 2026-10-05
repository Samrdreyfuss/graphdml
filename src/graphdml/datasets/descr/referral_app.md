# Referral App

**Question.** Does a promo coupon increase a user's spending, and does it raise spending among
their friends?

**Teaches.** How the *exposure map* defines what the peer effect means, and how hub-heavy
social graphs shrink the focal set.

| | |
|---|---|
| Nodes | 8,000 app users |
| Edges | friendships, grown by preferential attachment (a few influencers with many friends) |
| Treatment | `coupon` (0/1) |
| Outcome | `monthly_spend` in dollars |
| Covariates | `engagement_score`, `tenure_months`, `premium` |
| Exposure map | `"mean"`: the share of friends who received a coupon (0 to 1) |
| True direct effect | **+$12** for receiving a coupon |
| True peer effect | **+$40** going from 0% to 100% of friends with a coupon, i.e. **+$4 per 10%** |

**What confounds.** The company's targeting model sends coupons to engaged users *and* to users
whose friends are engaged, and engaged friends also raise spending directly.

**Choosing the exposure map.** The data were generated with the *share* of treated friends.
Refit with `exposure="sum"` and the peer coefficient changes meaning (dollars per additional
treated friend) and value, because the two maps disagree most for users with many friends.
The exposure map is an assumption you bring from domain knowledge: ask "would two coupons
among 4 friends matter as much as two among 400?"

```python
from graphdml import GraphDML, two_hop_test
from graphdml.datasets import make_referral_app

ds = make_referral_app()
for exposure in ["mean", "sum"]:
    m = GraphDML(exposure=exposure, random_state=0).fit(ds.data)
    print(exposure, m.summary_frame().round(2), sep="\n")
print(two_hop_test(GraphDML(exposure="mean", random_state=0), ds.data))
```

**Knobs.** `links_per_user` changes density; `confounding=0` removes targeting on friends.
