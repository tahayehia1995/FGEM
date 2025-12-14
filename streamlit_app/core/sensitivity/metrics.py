"""Sensitivity metrics: PRCC, Morris, Sobol (SALib)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _as_numeric_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = []
    for c in cols:
        X.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
    return np.vstack(X).T


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residualize y against X with intercept."""
    if X.size == 0:
        return y - np.nanmean(y)
    X2 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X2, y, rcond=None)
    return y - X2 @ beta


def compute_prcc(
    *,
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: str,
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Partial Rank Correlation Coefficients (PRCC) with bootstrap CI."""

    work = df[list(input_cols) + [output_col]].copy()
    for c in input_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[output_col] = pd.to_numeric(work[output_col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if len(work) < max(10, len(input_cols) + 3):
        raise ValueError("Not enough valid samples to compute PRCC.")

    # Rank-transform
    R = work.copy()
    for c in input_cols + [output_col]:
        R[c] = rankdata(R[c].to_numpy(dtype=float))

    X_all = _as_numeric_matrix(R, input_cols)
    y_all = R[output_col].to_numpy(dtype=float)

    rng = np.random.default_rng(int(seed))

    results: List[Dict[str, Any]] = []

    for i, p in enumerate(input_cols):
        others = [j for j in range(len(input_cols)) if j != i]
        X_others = X_all[:, others] if others else np.empty((len(R), 0))

        rx = _residualize(X_all[:, i], X_others)
        ry = _residualize(y_all, X_others)

        r = np.corrcoef(rx, ry)[0, 1]

        boots = []
        idx_all = np.arange(len(R))
        for _ in range(int(n_boot)):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            rx_b = rx[idx]
            ry_b = ry[idx]
            rb = np.corrcoef(rx_b, ry_b)[0, 1]
            boots.append(rb)
        boots_arr = np.array(boots, dtype=float)
        lo, hi = np.nanpercentile(boots_arr, [2.5, 97.5])

        results.append(
            {
                "parameter": p,
                "prcc": float(r),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n": int(len(R)),
            }
        )

    out = pd.DataFrame(results).sort_values("prcc", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


def compute_sobol(
    *,
    df: pd.DataFrame,
    design_meta: Dict[str, Any],
    output_col: str,
) -> pd.DataFrame:
    """Compute Sobol indices for the Sobol-tagged subset of df (requires complete runs)."""

    if "sobol_problem" not in design_meta:
        raise ValueError("Design metadata does not contain sobol_problem.")
    problem = design_meta["sobol_problem"]

    sub = df[df["__method"] == "Sobol"].copy()
    if "__sample_idx" in sub.columns:
        sub = sub.sort_values("__sample_idx")

    if "__status" in sub.columns:
        if (sub["__status"] != "ok").any():
            raise ValueError("Sobol analysis requires all Sobol cases to succeed (no missing Y).")

    y = pd.to_numeric(sub[output_col], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(y)):
        raise ValueError("Sobol analysis requires finite Y for all Sobol samples.")

    try:
        from SALib.analyze.sobol import analyze as sobol_analyze
    except Exception as e:
        raise ModuleNotFoundError(
            "SALib is required to compute Sobol indices. Install it (e.g., `pip install SALib`)."
        ) from e

    Si = sobol_analyze(
        problem,
        y,
        calc_second_order=bool(design_meta.get("sobol_calc_second_order", False)),
        print_to_console=False,
    )

    names = list(problem["names"])
    out_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        out_rows.append(
            {
                "parameter": name,
                "S1": float(Si["S1"][i]),
                "S1_conf": float(Si["S1_conf"][i]),
                "ST": float(Si["ST"][i]),
                "ST_conf": float(Si["ST_conf"][i]),
            }
        )
    out = pd.DataFrame(out_rows).sort_values("ST", ascending=False).reset_index(drop=True)
    return out


def compute_morris(
    *,
    df: pd.DataFrame,
    design_meta: Dict[str, Any],
    output_col: str,
) -> pd.DataFrame:
    """Compute Morris elementary effects (requires complete runs)."""

    if "morris_problem" not in design_meta:
        raise ValueError("Design metadata does not contain morris_problem.")
    problem = design_meta["morris_problem"]

    sub = df[df["__method"] == "Morris"].copy()
    if "__sample_idx" in sub.columns:
        sub = sub.sort_values("__sample_idx")

    if "__status" in sub.columns:
        if (sub["__status"] != "ok").any():
            raise ValueError("Morris analysis requires all Morris cases to succeed (no missing Y).")

    names = list(problem["names"])
    X = _as_numeric_matrix(sub, names)
    y = pd.to_numeric(sub[output_col], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(X)):
        raise ValueError("Morris analysis requires finite X and Y for all Morris samples.")

    try:
        from SALib.analyze.morris import analyze as morris_analyze
    except Exception as e:
        raise ModuleNotFoundError(
            "SALib is required to compute Morris indices. Install it (e.g., `pip install SALib`)."
        ) from e

    Si = morris_analyze(problem, X, y, print_to_console=False, num_levels=int(design_meta.get("settings", {}).get("morris_levels", 4)))

    out_rows: List[Dict[str, Any]] = []
    for i, name in enumerate(names):
        out_rows.append(
            {
                "parameter": name,
                "mu": float(Si["mu"][i]),
                "mu_star": float(Si["mu_star"][i]),
                "sigma": float(Si["sigma"][i]),
            }
        )
    out = pd.DataFrame(out_rows).sort_values("mu_star", ascending=False).reset_index(drop=True)
    return out

