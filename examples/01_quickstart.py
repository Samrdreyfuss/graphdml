# %% [markdown]
# # Quickstart: flu shots in a small town
#
# A runnable walkthrough (open as a notebook with Jupytext, or run as a script).
# 5,000 residents; getting a flu shot reduces your sick days (direct effect, -2.0) and each
# vaccinated neighbor reduces them further (peer effect, -0.5 per neighbor). Health-conscious
# residents cluster together, which confounds naive analyses.

# %%
from graphdml import GraphDML, placebo_test, two_hop_test
from graphdml.baselines import compare_methods
from graphdml.datasets import make_flu_town

ds = make_flu_town()
print(ds)
print(ds.DESCR)

# %% [markdown]
# ## Fit GraphDML
# The exposure map says how neighbors' treatments reach you: here, the number of vaccinated
# neighbors (`"sum"`). That is an assumption you bring, not something learned from data.

# %%
model = GraphDML(exposure="sum", random_state=0).fit(ds.data)
print(model.summary())

# %% [markdown]
# ## Why not just regress?
# Methods that ignore neighbors' covariates attribute the neighborhood's healthiness to the
# shot. Only estimators that adjust for network confounding cover the truth.

# %%
print(compare_methods(ds.data, exposure="sum", truth=ds.truth).round(2).to_string())

# %% [markdown]
# ## Stress-test the analysis
# These checks cannot prove the assumptions, but they flag common ways an analysis fails.

# %%
print(placebo_test(GraphDML(exposure="sum", random_state=0), ds.data, n_permutations=3))
print(two_hop_test(GraphDML(exposure="sum", random_state=0), ds.data))

# %% [markdown]
# ## Where the estimate comes from
# Only *focal* nodes, whose neighborhoods do not overlap, contribute to the final estimate.
# Their number is the effective sample size; they skew toward low-degree residents.

# %%
d = model.diagnostics_
print(f"focal nodes: {model.n_focal_} of {model.n_nodes_}")
print(f"mean degree: all {d['mean_degree_all']:.2f}, focal {d['mean_degree_focal']:.2f}")
print(model.residuals_.head())
