"""Difference-in-differences toolkit (ECON 5200 PS3), extending Lab 11's did_utils.py."""
from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

NEVER = 10**9

# Lab 11's functions, carried over
def two_by_two(df: pd.DataFrame, outcome: str, treated: str, post: str) -> float:
    """The canonical 2x2: difference of differences of the four group means."""
    g = df.groupby([treated, post])[outcome].mean()
    return (g[1, 1] - g[1, 0]) - (g[0, 1] - g[0, 0])


def att_by_cohort_period(df: pd.DataFrame, time: str, outcome: str,
                         cohort: str) -> pd.DataFrame:
    """ATT(g, t) for every cohort g and period t >= g.

    Controls are units NOT YET treated at t, never units already treated.
    The baseline is period g - 1, the last period before cohort g is treated.
    Returns one row per (g, t): the event time e = t - g, the estimate, and
    the number of treated units it rests on.
    """
    out = []
    cohorts = sorted(c for c in df[cohort].unique() if c < NEVER)
    for g in cohorts:
        tr = df[df[cohort] == g]
        for t in sorted(df.loc[df[time] >= g, time].unique()):
            co = df[df[cohort] > t]              # not yet treated at t
            if co.empty or (tr[time] == g - 1).sum() == 0:
                continue                         # no control left, or no baseline period
            d_tr = (tr.loc[tr[time] == t, outcome].mean()
                    - tr.loc[tr[time] == g - 1, outcome].mean())
            d_co = (co.loc[co[time] == t, outcome].mean()
                    - co.loc[co[time] == g - 1, outcome].mean())
            out.append({"g": g, "t": t, "e": t - g,
                        "att": d_tr - d_co, "n": int((tr[time] == t).sum())})
    return pd.DataFrame(out)


def overall_att(cells: pd.DataFrame) -> float:
    """Size-weighted average of the ATT(g, t) cells."""
    return float(np.average(cells["att"], weights=cells["n"]))


def event_study(cells: pd.DataFrame) -> pd.DataFrame:
    """Average the ATT(g, t) cells by event time e = t - g, weighted by size.

    Negative e would be pre-treatment periods. They are a diagnostic, not a
    test: estimates near zero are consistent with parallel trends and do not
    establish it.
    """
    rows = []
    for e, group in cells.groupby("e"):
        rows.append({"e": e, "att": np.average(group["att"], weights=group["n"])})
    return pd.DataFrame(rows)


def cluster_se_ratio(fit_iid, fit_clustered, param: str) -> float:
    """How many times larger the clustered standard error is than the iid one."""
    return float(fit_clustered.bse[param] / fit_iid.bse[param])

def clean_att(df: pd.DataFrame, outcome: str, cohort: str = 'g', time: str = 't', never: int = 10**9) -> tuple [pd.DataFrame, float]:
  """ATT(g, t) cells against not-yet-treated controls, and their size-weighted average
      Each cell compares cohort g's change from period g - 1 to period t with the same 
      change for every unit not yet treated at t. Returns a DF with one row per cell (g, t, att, n)
      and the n-weighted average of att.
  """

  out = []
  for g in sorted(df.loc[df[cohort] < never, cohort].unique()):
    treated = df[df[cohort] == g]
    for t in range(g, df[time].max() + 1):
      controls = df[df[cohort] > t]
      if controls.empty:
        continue
      d_tr = (treated.loc[treated[time] == t, outcome].mean()
              - treated.loc[treated[time] == g - 1, outcome].mean())
      d_co = (controls.loc[controls[time] == t, outcome].mean()
            - controls.loc[controls[time] == g -1, outcome].mean())
      n = (treated[time] == t).sum()
      out.append({"g": g, "t": t, "att": d_tr - d_co, "n": n})
    cells = pd.DataFrame(out)
  return cells, float(np.average(cells['att'], weights=cells['n']))
# your clustered-inference wrapper (Step 3.3)


def clustered_fits(df: pd.DataFrame, formula: str, param: str, clusters: dict[str, str]) -> pd.DataFrame:

  rows = []
  fit = smf.ols(formula, data=df).fit()
  rows.append({"fit": "default(iid)", "estimate": fit.params[param],
               "std_err": fit.bse[param], "p_value": fit.pvalues[param],
               "n_clusters": None})
  for label, column in clusters.items():
    fit = smf.ols(formula, data=df).fit(cov_type="cluster",
                                        cov_kwds={"groups": df[column]})
    rows.append({"fit": f"clustered by {label}", "estimate": fit.params[param],
                "std_err": fit.bse[param], "p_value": fit.pvalues[param]})
  return pd.DataFrame(rows)
  

# your bootstrap (Task 4.1)
"""Inference tools for difference-in-differences panels with few clusters."""




def wild_cluster_bootstrap_did(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit: str,
    time: str,
    cluster: str,
    tol: float = 1e-10,
) -> Dict[str, float]:
    """Wild cluster bootstrap-t for the treatment coefficient of a two-way DiD.

    Fits ``outcome ~ treatment + C(unit) + C(time)`` with cluster-robust
    standard errors, then tests H0: beta_treatment = 0 by the restricted
    ("null-imposed") wild cluster bootstrap of Cameron, Gelbach and Miller
    (2008). Rademacher weights are drawn once per cluster and *all* 2**G
    weight vectors are enumerated, so the bootstrap distribution is exact
    rather than simulated.

    Parameters
    ----------
    data:
        Long panel, one row per unit-period.
    outcome:
        Column name of the dependent variable.
    treatment:
        Column name of the treatment indicator whose coefficient is tested.
    unit:
        Column name of the cross-sectional identifier (absorbed as fixed
        effects).
    time:
        Column name of the period identifier (absorbed as fixed effects).
    cluster:
        Column name of the level at which treatment is assigned and errors
        are clustered.
    tol:
        Absolute tolerance used when comparing ``|t*|`` to ``|t|``, so that
        the all-positive weight vector -- which reproduces the actual data
        exactly in theory -- is not excluded by floating-point noise.

    Returns
    -------
    dict
        ``coef``, ``se`` and ``t`` for the treatment coefficient from the
        unrestricted cluster-robust fit; ``p_naive``, the cluster-robust
        p-value statsmodels reports for it; ``p_bootstrap``, the share of
        enumerated ``|t*|`` at least as large as ``|t|``; ``n_clusters``;
        ``n_replications`` (= 2**n_clusters); and ``min_p_attainable``, the
        smallest p-value the enumeration can produce.

    Raises
    ------
    ValueError
        If a required column is missing, if the cluster count is outside the
        range where full enumeration is feasible, or if treatment is not
        constant within cluster-period cells in a way that leaves it
        collinear with the fixed effects.
    """
    required = [outcome, treatment, unit, time, cluster]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"missing column(s): {missing}")

    df = data.loc[:, required].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("no complete rows after dropping missing values")

    codes, _ = pd.factorize(df[cluster], sort=True)
    codes = np.asarray(codes, dtype=int)
    n_clusters = int(codes.max()) + 1
    if not 2 <= n_clusters <= 20:
        raise ValueError(
            f"full enumeration needs between 2 and 20 clusters; got "
            f"{n_clusters} (2**{n_clusters} weight vectors)"
        )
    groups = codes

    fe = f"C({unit}) + C({time})"
    unrestricted_formula = f"{outcome} ~ {treatment} + {fe}"
    restricted_formula = f"{outcome} ~ {fe}"

    # Unrestricted model: design matrix is reused for every bootstrap fit.
    model = smf.ols(unrestricted_formula, data=df)
    exog = np.asarray(model.exog, dtype=float)
    try:
        k_treat = list(model.exog_names).index(treatment)
    except ValueError as err:  # pragma: no cover - defensive
        raise ValueError(f"{treatment!r} not in the design matrix") from err

    cov_kwds = {"groups": groups}
    actual = model.fit(cov_type="cluster", cov_kwds=cov_kwds)
    coef = float(actual.params.iloc[k_treat])
    se = float(actual.bse.iloc[k_treat])
    t_actual = float(actual.tvalues.iloc[k_treat])
    p_naive = float(actual.pvalues.iloc[k_treat])
    if not np.isfinite(t_actual):
        raise ValueError(
            f"{treatment!r} is collinear with the {unit}/{time} fixed effects"
        )

    # Step 1: restricted fit -- the treatment coefficient forced to zero.
    restricted = smf.ols(restricted_formula, data=df).fit()
    y_tilde = np.asarray(restricted.fittedvalues, dtype=float)
    u_tilde = np.asarray(restricted.resid, dtype=float)

    # Step 2: all 2**G Rademacher vectors, one weight per cluster, expanded
    # to one weight per observation via the cluster map.
    draws = np.arange(2**n_clusters)
    signs = 1.0 - 2.0 * ((draws[:, None] >> np.arange(n_clusters)) & 1)
    weights = signs[:, codes]

    t_star = np.empty(len(draws), dtype=float)
    for r in range(len(draws)):
        # Step 3: synthetic outcome under the null.
        y_star = y_tilde + weights[r] * u_tilde
        # Step 4: re-estimate the unrestricted model, keep the clustered t.
        fit = sm.OLS(y_star, exog).fit(cov_type="cluster", cov_kwds=cov_kwds)
        t_star[r] = fit.tvalues[k_treat]

    # Step 5: symmetric (two-sided) bootstrap p-value.
    p_bootstrap = float(
        np.mean(np.abs(t_star) >= np.abs(t_actual) - tol)
    )

    return {
        "coef": coef,
        "se": se,
        "t": t_actual,
        "p_naive": p_naive,
        "p_bootstrap": p_bootstrap,
        "n_clusters": float(n_clusters),
        "n_replications": float(len(draws)),
        "min_p_attainable": 2.0 / len(draws),
    }

if __name__ == "__main__":
    ...   # passing assertions
