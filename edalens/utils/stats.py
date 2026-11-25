"""Statistics behind the Statistics page; Streamlit-free so they can be unit-tested.

- ``summary_table``: one row per column (type, missing, unique, mean, median, sd,
  skew, outliers) and ``duplicate_rows``.
- ``compare_groups``: a numeric column across the levels of a categorical one
  (t-test / ANOVA or Mann-Whitney / Kruskal-Wallis, with an effect size).
- ``chi_square``: two categorical columns, with Cramér's V.
- ``correlations``: Pearson / Spearman / Kendall with p-values, top pairs, and
  ``cramers_v_matrix`` for categorical columns.
- ``outlier_mask``: rows flagged by the IQR rule or by z-scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats

ALPHA = 0.05
OUTLIER_METHODS = ("IQR", "Z-score")
DEFAULT_THRESHOLDS = {"IQR": 1.5, "Z-score": 3.0}
CORRELATION_METHODS = ("pearson", "spearman", "kendall")
PARAMETRIC = "Parametric (t-test / ANOVA)"
NONPARAMETRIC = "Rank-based (Mann-Whitney / Kruskal-Wallis)"
GROUP_TESTS = (PARAMETRIC, NONPARAMETRIC)


def _numeric(series: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def column_kind(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if _numeric(series):
        return "numeric"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


# -- Summary ----------------------------------------------------------------------
def outlier_mask(
    frame: pd.DataFrame,
    columns: Sequence[str],
    method: str = "IQR",
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """True where a value is an outlier in its column (one boolean column per column).

    IQR: below Q1 - k·IQR or above Q3 + k·IQR (k = 1.5 by default).
    Z-score: |value - mean| / sd above k (3 by default). Missing values are never outliers.
    """
    k = DEFAULT_THRESHOLDS[method] if threshold is None else threshold
    flags = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if method == "IQR":
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            flags[column] = (values < q1 - k * spread) | (values > q3 + k * spread)
        else:
            sd = values.std()
            flags[column] = (values - values.mean()).abs() / sd > k if sd and sd > 0 else values * False
        flags[column] = flags[column].fillna(False).astype(bool)
    return pd.DataFrame(flags, index=frame.index)


def summary_table(frame: pd.DataFrame, outlier_method: str = "IQR") -> pd.DataFrame:
    """One row per column: type, missing count and %, unique values, and for numbers
    mean, median, sd, min, max, skew and the outlier count (``outlier_method``)."""
    rows = []
    numeric = [c for c in frame.columns if _numeric(frame[c])]
    outliers = outlier_mask(frame, numeric, outlier_method).sum() if numeric else pd.Series(dtype=int)
    n = len(frame)
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        row = {
            "Column": str(column),
            "Type": column_kind(series),
            "Missing": missing,
            "Missing %": round(100 * missing / n, 2) if n else 0.0,
            "Unique": int(series.nunique(dropna=True)),
        }
        if column in numeric:
            values = series.astype("float64")
            row.update(
                {
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "SD": values.std(),
                    "Min": values.min(),
                    "Max": values.max(),
                    "Skew": values.skew(),
                    "Outliers": int(outliers[column]),
                }
            )
        else:
            top = series.mode(dropna=True)
            row["Most common"] = str(top.iloc[0]) if not top.empty else None
        rows.append(row)
    columns = ["Column", "Type", "Missing", "Missing %", "Unique", "Mean", "Median", "SD",
               "Min", "Max", "Skew", "Outliers", "Most common"]
    table = pd.DataFrame(rows)
    return table.reindex(columns=[c for c in columns if c in table.columns])


def duplicate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Every row that has an identical copy elsewhere (all copies shown)."""
    return frame[frame.duplicated(keep=False)]


# -- Group comparisons -----------------------------------------------------------------
@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect: float
    effect_label: str
    groups: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return bool(self.p_value < ALPHA)

    def sentence(self) -> str:
        verdict = "a significant difference" if self.significant else "no significant difference"
        return (
            f"{self.test}: statistic = {self.statistic:.3g}, {format_p(self.p_value)} → "
            f"{verdict} at α = {ALPHA}. Effect size {self.effect_name} = {self.effect:.3g} "
            f"({self.effect_label})."
        )


def format_p(p: float) -> str:
    """'p < 0.001' or 'p = 0.034'."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _label(value: float, cuts: Sequence[float]) -> str:
    """'negligible', 'small', 'medium' or 'large' by the conventional cut-offs."""
    value = abs(value)
    for name, cut in zip(("negligible", "small", "medium"), cuts):
        if value < cut:
            return name
    return "large"


def compare_groups(
    frame: pd.DataFrame, value: str, group: str, test: str = PARAMETRIC, min_size: int = 2
) -> TestResult:
    """Does ``value`` differ across the levels of ``group``?

    Two groups: Welch's t-test with Cohen's d (parametric) or Mann-Whitney U with
    the rank-biserial correlation. More: one-way ANOVA with eta² or Kruskal-Wallis
    with epsilon². Groups with fewer than ``min_size`` values are left out.
    """
    data = frame[[value, group]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    samples = {str(k): g[value].to_numpy(float) for k, g in data.groupby(group, observed=True, sort=False)}
    notes = []
    small = [k for k, v in samples.items() if len(v) < min_size]
    if small:
        notes.append(f"Left out (fewer than {min_size} values): {', '.join(small)}.")
    samples = {k: v for k, v in samples.items() if len(v) >= min_size}
    if len(samples) < 2:
        raise ValueError(f"Comparing needs at least two groups of {group} with {min_size}+ values of {value}.")
    groups = pd.DataFrame(
        {
            group: list(samples),
            "n": [len(v) for v in samples.values()],
            "Mean": [v.mean() for v in samples.values()],
            "SD": [v.std(ddof=1) for v in samples.values()],
            "Median": [np.median(v) for v in samples.values()],
        }
    )
    arrays = list(samples.values())
    n_total = sum(len(v) for v in arrays)
    if len(arrays) == 2:
        a, b = arrays
        if test == PARAMETRIC:
            result = stats.ttest_ind(a, b, equal_var=False)
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0
            return TestResult("Welch's t-test", result.statistic, result.pvalue, "Cohen's d", d,
                              _label(d, (0.2, 0.5, 0.8)), groups, notes)
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = 1 - 2 * result.statistic / (len(a) * len(b))
        return TestResult("Mann-Whitney U", result.statistic, result.pvalue, "rank-biserial r", r,
                          _label(r, (0.1, 0.3, 0.5)), groups, notes)
    k = len(arrays)
    if test == PARAMETRIC:
        result = stats.f_oneway(*arrays)
        grand = np.concatenate(arrays).mean()
        between = sum(len(v) * (v.mean() - grand) ** 2 for v in arrays)
        total = ((np.concatenate(arrays) - grand) ** 2).sum()
        eta = between / total if total > 0 else 0.0
        return TestResult("One-way ANOVA", result.statistic, result.pvalue, "η²", eta,
                          _label(eta, (0.01, 0.06, 0.14)), groups, notes)
    result = stats.kruskal(*arrays)
    epsilon = (result.statistic - k + 1) / (n_total - k) if n_total > k else 0.0
    return TestResult("Kruskal-Wallis H", result.statistic, result.pvalue, "ε²", epsilon,
                      _label(epsilon, (0.01, 0.06, 0.14)), groups, notes)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def chi_square(frame: pd.DataFrame, a: str, b: str) -> tuple[TestResult, pd.DataFrame]:
    """Chi-square test of independence of two categorical columns, with Cramér's V.
    Returns the result and the contingency table."""
    data = frame[[a, b]].dropna()
    table = pd.crosstab(data[a], data[b])
    if min(table.shape) < 2:
        raise ValueError(f"The chi-square test needs at least two values in both {a} and {b}.")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = cramers_v(data[a], data[b])
    notes = []
    if (expected < 5).mean() > 0.2:
        notes.append("More than 20% of the expected counts are below 5, so the p-value is approximate.")
    k = min(table.shape) - 1
    cuts = tuple(c / np.sqrt(k) for c in (0.1, 0.3, 0.5))
    result = TestResult(f"Chi-square (df = {dof})", chi2, p, "Cramér's V", v, _label(v, cuts), notes=notes)
    return result, table


# -- Correlations -----------------------------------------------------------------------
def correlations(frame: pd.DataFrame, columns: Sequence[str], method: str = "pearson") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation matrix and the matching matrix of p-values (pairwise complete rows)."""
    test = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    columns = list(columns)
    r = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    p = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    for x, y in combinations(columns, 2):
        pair = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
            value, pvalue = np.nan, np.nan
        else:
            value, pvalue = test(pair[x], pair[y])[:2]
        r.loc[x, y] = r.loc[y, x] = value
        p.loc[x, y] = p.loc[y, x] = pvalue
    return r, p


def top_pairs(r: pd.DataFrame, p: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """The ``limit`` strongest pairs (by |r|), with p-values."""
    rows = [
        {"Variable 1": x, "Variable 2": y, "r": r.loc[x, y], "p-value": p.loc[x, y]}
        for x, y in combinations(r.columns, 2)
        if pd.notna(r.loc[x, y])
    ]
    table = pd.DataFrame(rows, columns=["Variable 1", "Variable 2", "r", "p-value"])
    return table.reindex(table["r"].abs().sort_values(ascending=False).index).head(limit).reset_index(drop=True)


def cramers_v_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    columns = list(columns)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for a, b in combinations(columns, 2):
        data = frame[[a, b]].dropna()
        matrix.loc[a, b] = matrix.loc[b, a] = cramers_v(data[a], data[b]) if len(data) else np.nan
    return matrix
