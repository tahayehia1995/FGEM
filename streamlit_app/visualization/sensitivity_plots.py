"""Scientific plotting utilities for sensitivity analysis results."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def plot_prcc_bar_plotly(prcc_df: pd.DataFrame, title: str = "PRCC (with 95% CI)"):
    import plotly.graph_objects as go

    df = prcc_df.copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values("prcc", key=lambda s: s.abs(), ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["parameter"],
            x=df["prcc"],
            orientation="h",
            error_x=dict(
                type="data",
                symmetric=False,
                array=df["ci_high"] - df["prcc"],
                arrayminus=df["prcc"] - df["ci_low"],
                thickness=1.5,
            ),
        )
    )
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="<b>PRCC</b>",
        yaxis_title="<b>Parameter</b>",
        height=max(420, 28 * len(df)),
        margin=dict(l=120, r=40, t=60, b=50),
    )
    fig.update_xaxes(zeroline=True, zerolinewidth=2)
    return fig


def plot_prcc_bar_matplotlib(prcc_df: pd.DataFrame, title: str = "PRCC (with 95% CI)"):
    import matplotlib.pyplot as plt

    df = prcc_df.copy()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * max(1, len(df)))))
    if df.empty:
        ax.set_title(title, fontweight="bold")
        return fig
    df = df.sort_values("prcc", key=lambda s: s.abs(), ascending=True)
    y = np.arange(len(df))
    x = df["prcc"].to_numpy(dtype=float)
    xerr_low = x - df["ci_low"].to_numpy(dtype=float)
    xerr_high = df["ci_high"].to_numpy(dtype=float) - x
    ax.barh(y, x, xerr=[xerr_low, xerr_high], alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.axvline(0, color="k", linewidth=1)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("PRCC", fontweight="bold")
    ax.set_ylabel("Parameter", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sobol_indices_plotly(sobol_df: pd.DataFrame, title: str = "Sobol indices (S1, ST)"):
    import plotly.graph_objects as go

    df = sobol_df.copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values("ST", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["parameter"],
            x=df["S1"],
            name="S1",
            orientation="h",
            error_x=dict(type="data", array=df["S1_conf"], thickness=1.2),
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Bar(
            y=df["parameter"],
            x=df["ST"],
            name="ST",
            orientation="h",
            error_x=dict(type="data", array=df["ST_conf"], thickness=1.2),
            opacity=0.55,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title=f"<b>{title}</b>",
        xaxis_title="<b>Sensitivity index</b>",
        yaxis_title="<b>Parameter</b>",
        height=max(420, 28 * len(df)),
        margin=dict(l=120, r=40, t=60, b=50),
    )
    fig.update_xaxes(range=[0, max(1e-6, float(df[["S1", "ST"]].max().max()) * 1.15)])
    return fig


def plot_sobol_indices_matplotlib(sobol_df: pd.DataFrame, title: str = "Sobol indices (S1, ST)"):
    import matplotlib.pyplot as plt

    df = sobol_df.copy()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * max(1, len(df)))))
    if df.empty:
        ax.set_title(title, fontweight="bold")
        return fig
    df = df.sort_values("ST", ascending=True)
    y = np.arange(len(df))
    ax.barh(y, df["ST"], xerr=df["ST_conf"], alpha=0.5, label="ST")
    ax.barh(y, df["S1"], xerr=df["S1_conf"], alpha=0.85, label="S1")
    ax.set_yticks(y)
    ax.set_yticklabels(df["parameter"])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Sensitivity index", fontweight="bold")
    ax.set_ylabel("Parameter", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_morris_scatter_plotly(morris_df: pd.DataFrame, title: str = "Morris screening (μ* vs σ)"):
    import plotly.express as px

    df = morris_df.copy()
    if df.empty:
        return px.scatter()
    fig = px.scatter(
        df,
        x="mu_star",
        y="sigma",
        text="parameter",
        hover_data=["mu", "mu_star", "sigma"],
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="<b>μ*</b>",
        yaxis_title="<b>σ</b>",
        height=520,
    )
    return fig


def plot_morris_scatter_matplotlib(morris_df: pd.DataFrame, title: str = "Morris screening (μ* vs σ)"):
    import matplotlib.pyplot as plt

    df = morris_df.copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty:
        ax.set_title(title, fontweight="bold")
        return fig
    ax.scatter(df["mu_star"], df["sigma"], alpha=0.8)
    for _, r in df.iterrows():
        ax.annotate(str(r["parameter"]), (r["mu_star"], r["sigma"]), fontsize=9)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("μ*", fontweight="bold")
    ax.set_ylabel("σ", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_distribution_plotly(df: pd.DataFrame, col: str, title: Optional[str] = None, nbins: int = 40):
    import plotly.express as px

    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fig = px.histogram(s, nbins=nbins, marginal="box")
    fig.update_layout(
        title=f"<b>{title or f'Distribution of {col}'}</b>",
        xaxis_title=f"<b>{col}</b>",
        yaxis_title="<b>Count</b>",
        height=420,
    )
    return fig


def plot_corr_heatmap_plotly(df: pd.DataFrame, cols: Sequence[str], title: str = "Correlation heatmap (Spearman)"):
    import plotly.express as px

    # Guard against duplicate column names (df['x'] returns DataFrame when duplicated)
    cols_unique: List[str] = []
    seen = set()
    for c in cols:
        if c not in seen:
            cols_unique.append(c)
            seen.add(c)

    work = df[list(cols_unique)].copy()
    for c in cols_unique:
        series_or_df = work[c]
        if isinstance(series_or_df, pd.DataFrame):
            # duplicated column labels in the underlying df; pick the first
            series_or_df = series_or_df.iloc[:, 0]
        work[c] = pd.to_numeric(series_or_df, errors="coerce")
    corr = work.corr(method="spearman")
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig.update_layout(title=f"<b>{title}</b>", height=max(450, 25 * len(cols_unique)))
    return fig


def plot_scatter_matrix_plotly(df: pd.DataFrame, cols: Sequence[str], title: str = "Scatter matrix"):
    import plotly.express as px

    work = df[list(cols)].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    fig = px.scatter_matrix(work, dimensions=list(cols))
    fig.update_layout(title=f"<b>{title}</b>", height=700)
    return fig


def plot_cdf_plotly(df: pd.DataFrame, col: str, title: Optional[str] = None):
    import plotly.express as px

    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    if s.empty:
        return px.line()
    y = np.linspace(0.0, 1.0, len(s))
    fig = px.line(x=s.to_numpy(), y=y)
    fig.update_layout(
        title=f"<b>{title or f'Empirical CDF of {col}'}</b>",
        xaxis_title=f"<b>{col}</b>",
        yaxis_title="<b>CDF</b>",
        height=420,
    )
    return fig


def plot_violin_plotly(df: pd.DataFrame, col: str, by: Optional[str] = None, title: Optional[str] = None):
    import plotly.express as px

    work = df.copy()
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    if by and by in work.columns:
        fig = px.violin(work, y=col, x=by, box=True, points="outliers")
    else:
        fig = px.violin(work, y=col, box=True, points="outliers")
    fig.update_layout(title=f"<b>{title or f'Violin plot of {col}'}</b>", height=420)
    return fig


def plot_tornado_oat_plotly(df: pd.DataFrame, output_col: str, title: Optional[str] = None):
    """Tornado-like plot for OAT runs: effect range per parameter."""
    import plotly.express as px

    sub = df[df["__method"] == "OAT"].copy()
    sub = sub[sub.get("__status", "ok") == "ok"]
    if "__param" not in sub.columns:
        return px.bar()

    sub[output_col] = pd.to_numeric(sub[output_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[output_col])
    if sub.empty:
        return px.bar()

    agg = sub.groupby("__param")[output_col].agg(["min", "max", "median"]).reset_index()
    agg["range"] = agg["max"] - agg["min"]
    agg = agg.sort_values("range", ascending=True)
    fig = px.bar(agg, x="range", y="__param", orientation="h")
    fig.update_layout(
        title=f"<b>{title or f'OAT tornado (range) for {output_col}'}</b>",
        xaxis_title=f"<b>Range of {output_col}</b>",
        yaxis_title="<b>Parameter</b>",
        height=max(420, 26 * len(agg)),
        margin=dict(l=140, r=40, t=60, b=50),
    )
    return fig


def plot_tornado_oat_matplotlib(df: pd.DataFrame, output_col: str, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    sub = df[df["__method"] == "OAT"].copy()
    sub = sub[sub.get("__status", "ok") == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5))
    if sub.empty or "__param" not in sub.columns:
        ax.set_title(title or "OAT tornado", fontweight="bold")
        return fig
    sub[output_col] = pd.to_numeric(sub[output_col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[output_col])
    agg = sub.groupby("__param")[output_col].agg(["min", "max"]).reset_index()
    agg["range"] = agg["max"] - agg["min"]
    agg = agg.sort_values("range", ascending=True)
    y = np.arange(len(agg))
    ax.barh(y, agg["range"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(agg["__param"])
    ax.set_title(title or f"OAT tornado (range) for {output_col}", fontweight="bold")
    ax.set_xlabel(f"Range of {output_col}", fontweight="bold")
    ax.set_ylabel("Parameter", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_spider_oat_plotly(df: pd.DataFrame, param: str, output_col: str, title: Optional[str] = None):
    """Spider plot for a single OAT parameter sweep."""
    import plotly.express as px

    sub = df[(df["__method"] == "OAT") & (df.get("__param") == param)].copy()
    sub = sub[sub.get("__status", "ok") == "ok"]
    if sub.empty or param not in sub.columns:
        return px.line()

    x = pd.to_numeric(sub[param], errors="coerce")
    y = pd.to_numeric(sub[output_col], errors="coerce")
    work = pd.DataFrame({param: x, output_col: y}).replace([np.inf, -np.inf], np.nan).dropna()
    work = work.sort_values(param)
    fig = px.line(work, x=param, y=output_col, markers=True)
    fig.update_layout(
        title=f"<b>{title or f'OAT response: {output_col} vs {param}'}</b>",
        xaxis_title=f"<b>{param}</b>",
        yaxis_title=f"<b>{output_col}</b>",
        height=420,
    )
    return fig


def plot_spider_oat_matplotlib(df: pd.DataFrame, param: str, output_col: str, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    sub = df[(df["__method"] == "OAT") & (df.get("__param") == param)].copy()
    sub = sub[sub.get("__status", "ok") == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4))
    if sub.empty or param not in sub.columns:
        ax.set_title(title or "OAT response", fontweight="bold")
        return fig
    x = pd.to_numeric(sub[param], errors="coerce")
    y = pd.to_numeric(sub[output_col], errors="coerce")
    work = pd.DataFrame({param: x, output_col: y}).replace([np.inf, -np.inf], np.nan).dropna().sort_values(param)
    ax.plot(work[param], work[output_col], marker="o")
    ax.set_title(title or f"OAT response: {output_col} vs {param}", fontweight="bold")
    ax.set_xlabel(param, fontweight="bold")
    ax.set_ylabel(output_col, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_response_surface_plotly(df: pd.DataFrame, x: str, y: str, z: str, title: Optional[str] = None, contour: bool = False):
    """2D response surface from Grid/Factorial subset."""
    import plotly.express as px

    sub = df[df["__method"] == "Grid/Factorial"].copy()
    sub = sub[sub.get("__status", "ok") == "ok"]
    if sub.empty:
        return px.imshow(np.zeros((1, 1)))

    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub[z] = pd.to_numeric(sub[z], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y, z])
    if sub.empty:
        return px.imshow(np.zeros((1, 1)))

    # pivot to a grid (if not full, we use mean aggregator)
    pivot = sub.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)

    if contour:
        fig = px.contour(
            x=pivot.columns.to_numpy(dtype=float),
            y=pivot.index.to_numpy(dtype=float),
            z=pivot.to_numpy(dtype=float),
            labels={"x": x, "y": y, "color": z},
        )
    else:
        fig = px.imshow(
            pivot.to_numpy(dtype=float),
            x=pivot.columns.to_numpy(dtype=float),
            y=pivot.index.to_numpy(dtype=float),
            aspect="auto",
            origin="lower",
            color_continuous_scale="Viridis",
            labels={"x": x, "y": y, "color": z},
        )
    fig.update_layout(
        title=f"<b>{title or f'Response surface: {z} vs ({x}, {y})'}</b>",
        xaxis_title=f"<b>{x}</b>",
        yaxis_title=f"<b>{y}</b>",
        height=520,
    )
    return fig


def plot_response_surface_matplotlib(df: pd.DataFrame, x: str, y: str, z: str, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    sub = df[df["__method"] == "Grid/Factorial"].copy()
    sub = sub[sub.get("__status", "ok") == "ok"]
    fig, ax = plt.subplots(figsize=(7, 5))
    if sub.empty:
        ax.set_title(title or "Response surface", fontweight="bold")
        return fig
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub[z] = pd.to_numeric(sub[z], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y, z])
    pivot = sub.pivot_table(index=y, columns=x, values=z, aggfunc="mean").sort_index().sort_index(axis=1)
    im = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto")
    ax.set_title(title or f"Response surface: {z} vs ({x}, {y})", fontweight="bold")
    ax.set_xlabel(x, fontweight="bold")
    ax.set_ylabel(y, fontweight="bold")
    fig.colorbar(im, ax=ax, label=z)
    fig.tight_layout()
    return fig

