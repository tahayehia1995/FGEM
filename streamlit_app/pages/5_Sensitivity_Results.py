"""Sensitivity analysis results + visualization page for FGEM Streamlit app."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.core.sensitivity.metrics import compute_morris, compute_prcc, compute_sobol
from streamlit_app.visualization.sensitivity_plots import (
    plot_cdf_plotly,
    plot_corr_heatmap_plotly,
    plot_distribution_plotly,
    plot_morris_scatter_matplotlib,
    plot_morris_scatter_plotly,
    plot_prcc_bar_matplotlib,
    plot_prcc_bar_plotly,
    plot_response_surface_matplotlib,
    plot_response_surface_plotly,
    plot_scatter_matrix_plotly,
    plot_sobol_indices_matplotlib,
    plot_sobol_indices_plotly,
    plot_spider_oat_matplotlib,
    plot_spider_oat_plotly,
    plot_tornado_oat_matplotlib,
    plot_tornado_oat_plotly,
    plot_violin_plotly,
)


st.set_page_config(page_title="Sensitivity Results - FGEM", page_icon="📉", layout="wide")

st.title("📉 Sensitivity Results & Visualizations")
st.markdown("Explore sensitivity outputs with scientific plots, filtering, and exports.")

if "sa_run" not in st.session_state or st.session_state.sa_run is None:
    st.warning("⚠️ No sensitivity run found. Please run a sensitivity batch first.")
    if st.button("Go to Sensitivity Analysis"):
        st.switch_page("pages/4_Sensitivity_Analysis.py")
    st.stop()

sa_run: Dict[str, Any] = st.session_state.sa_run
design: Dict[str, Any] = sa_run.get("design", {}) or {}
df: pd.DataFrame = sa_run.get("results", pd.DataFrame()).copy()
outputs: List[str] = list(sa_run.get("outputs", []))

if df.empty:
    st.warning("⚠️ Sensitivity results are empty.")
    st.stop()

design_meta = (design.get("meta") or {}) if isinstance(design, dict) else {}
methods = sorted(df["__method"].dropna().unique().tolist()) if "__method" in df.columns else []

st.header("📌 Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total cases", f"{len(df):,}")
with col2:
    ok = int((df.get("__status") == "ok").sum()) if "__status" in df.columns else len(df)
    st.metric("Succeeded", f"{ok:,}")
with col3:
    err = int((df.get("__status") == "error").sum()) if "__status" in df.columns else 0
    st.metric("Failed", f"{err:,}")
with col4:
    st.metric("Methods", ", ".join(methods) if methods else "—")

with st.expander("Design metadata", expanded=False):
    st.json(design_meta)

st.header("🎛️ Plot backend")
plot_backend = st.radio("Backend", options=["Plotly (interactive)", "Matplotlib (static)"], horizontal=True)

st.header("🔎 Filters")

f1, f2, f3, f4 = st.columns([1.0, 1.0, 1.0, 1.0])
with f1:
    method_filter = st.multiselect("Method", options=methods, default=methods)
with f2:
    status_filter = st.multiselect("Status", options=["ok", "error"], default=["ok", "error"] if err else ["ok"])
with f3:
    output_for_filter = st.selectbox("Filter by output (optional)", options=["None"] + outputs)
with f4:
    if output_for_filter != "None":
        q = st.slider("Keep percentile range", 0, 100, (1, 99))
    else:
        q = None

df_f = df.copy()
if "__method" in df_f.columns and method_filter:
    df_f = df_f[df_f["__method"].isin(method_filter)]
if "__status" in df_f.columns and status_filter:
    df_f = df_f[df_f["__status"].isin(status_filter)]
if output_for_filter != "None":
    y = pd.to_numeric(df_f[output_for_filter], errors="coerce")
    lo = np.nanpercentile(y, q[0]) if q else None
    hi = np.nanpercentile(y, q[1]) if q else None
    if lo is not None and hi is not None:
        df_f = df_f[(y >= lo) & (y <= hi)]

st.subheader("Results table")
st.dataframe(df_f.head(500), use_container_width=True)

tabs = st.tabs(
    [
        "OAT/Grid",
        "Distributions",
        "Correlations",
        "PRCC",
        "Morris",
        "Sobol",
        "Export",
    ]
)

with tabs[0]:
    st.subheader("OAT and Grid/Factorial visualizations")
    ok_df = df_f[df_f.get("__status", "ok") == "ok"].copy()
    if "OAT" in methods:
        st.markdown("**OAT (One-at-a-time)**")
        if outputs:
            out_col = st.selectbox("OAT output", options=outputs, key="oat_out")
            if plot_backend.startswith("Matplotlib"):
                fig = plot_tornado_oat_matplotlib(ok_df, out_col)
                st.pyplot(fig)
            else:
                fig = plot_tornado_oat_plotly(ok_df, out_col)
                st.plotly_chart(fig, use_container_width=True)

            if "__param" in ok_df.columns:
                params = sorted(ok_df.loc[ok_df["__method"] == "OAT", "__param"].dropna().unique().tolist())
                if params:
                    p = st.selectbox("OAT parameter (spider/response)", options=params, key="oat_param")
                    if plot_backend.startswith("Matplotlib"):
                        fig2 = plot_spider_oat_matplotlib(ok_df, p, out_col)
                        st.pyplot(fig2)
                    else:
                        fig2 = plot_spider_oat_plotly(ok_df, p, out_col)
                        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No OAT cases in this run.")

    st.markdown("---")
    if "Grid/Factorial" in methods:
        st.markdown("**Grid/Factorial (2D response surfaces)**")
        ok_g = ok_df[ok_df["__method"] == "Grid/Factorial"].copy() if "__method" in ok_df.columns else ok_df.copy()
        if outputs and not ok_g.empty:
            out_col = st.selectbox("Grid output (z)", options=outputs, key="grid_out")
            # candidate numeric parameters present in grid data
            cand = [c for c in ok_g.columns if (not c.startswith("__")) and (c not in outputs)]
            cand = [c for c in cand if pd.api.types.is_numeric_dtype(pd.to_numeric(ok_g[c], errors="coerce"))]
            if len(cand) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    x = st.selectbox("X", options=cand, index=0, key="grid_x")
                with c2:
                    y = st.selectbox("Y", options=[c for c in cand if c != x], index=0, key="grid_y")
                contour = st.checkbox("Contour instead of heatmap", value=False)
                if plot_backend.startswith("Matplotlib"):
                    fig = plot_response_surface_matplotlib(ok_df, x=x, y=y, z=out_col)
                    st.pyplot(fig)
                else:
                    fig = plot_response_surface_plotly(ok_df, x=x, y=y, z=out_col, contour=contour)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric swept parameters in Grid/Factorial to build a response surface.")
    else:
        st.info("No Grid/Factorial cases in this run.")

with tabs[1]:
    st.subheader("Output distributions")
    if not outputs:
        st.info("No outputs selected in the sensitivity run.")
    else:
        out_col = st.selectbox("Output", options=outputs, index=0)
        nb = st.number_input("Bins", min_value=10, value=40, step=5)
        ok_df = df_f[df_f.get("__status", "ok") == "ok"]
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_distribution_plotly(ok_df, out_col, nbins=int(nb))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = plot_cdf_plotly(ok_df, out_col)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Box/violin**")
        by = "__method" if "__method" in ok_df.columns else None
        fig3 = plot_violin_plotly(ok_df, out_col, by=by, title=f"{out_col} by method" if by else None)
        st.plotly_chart(fig3, use_container_width=True)

with tabs[2]:
    st.subheader("Correlation diagnostics")
    ok_df = df_f[df_f.get("__status", "ok") == "ok"].copy()
    if not outputs:
        st.info("No outputs selected.")
    else:
        out_col = st.selectbox("Output (for pairwise)", options=outputs, key="corr_out")
        # choose a small set of numeric inputs present in df
        candidate_inputs = [c for c in ok_df.columns if (not c.startswith("__")) and pd.api.types.is_numeric_dtype(ok_df[c])]
        default_inputs = candidate_inputs[: min(8, len(candidate_inputs))]
        sel_inputs = st.multiselect("Inputs", options=candidate_inputs, default=default_inputs)
        # Deduplicate while preserving order (avoids pandas returning a DataFrame for work[c])
        cols = list(dict.fromkeys(list(sel_inputs) + [out_col]))
        if len(cols) >= 2:
            fig = plot_corr_heatmap_plotly(ok_df, cols, title="Correlation heatmap (Spearman)")
            st.plotly_chart(fig, use_container_width=True)
            if st.checkbox("Show scatter matrix (can be heavy)", value=False):
                fig2 = plot_scatter_matrix_plotly(ok_df[cols].dropna(), cols, title="Scatter matrix")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Select at least 1 input and 1 output.")

with tabs[3]:
    st.subheader("PRCC (Partial Rank Correlation)")
    if "MonteCarlo" not in methods and "PRCC" not in (design_meta.get("methods") or []):
        st.info("No Monte Carlo / PRCC design found in this run.")
    else:
        ok_df = df[df.get("__method") == "MonteCarlo"].copy() if "__method" in df.columns else df.copy()
        ok_df = ok_df[ok_df.get("__status", "ok") == "ok"].copy()
        out_col = st.selectbox("Output", options=outputs, key="prcc_out")
        # numeric inputs: all non-meta numeric columns excluding outputs
        input_candidates = [c for c in ok_df.columns if (not c.startswith("__")) and (c not in outputs)]
        input_candidates = [c for c in input_candidates if pd.api.types.is_numeric_dtype(pd.to_numeric(ok_df[c], errors="coerce"))]
        default_inputs = input_candidates[: min(12, len(input_candidates))]
        sel_inputs = st.multiselect("Inputs (numeric)", options=input_candidates, default=default_inputs)
        n_boot = st.number_input("Bootstrap samples", min_value=50, value=300, step=50)
        seed = st.number_input("Bootstrap seed", min_value=0, value=123, step=1, key="prcc_seed")
        if st.button("Compute PRCC", type="primary"):
            with st.spinner("Computing PRCC..."):
                prcc_df = compute_prcc(df=ok_df, input_cols=sel_inputs, output_col=out_col, n_boot=int(n_boot), seed=int(seed))
            st.dataframe(prcc_df, use_container_width=True)
            if plot_backend.startswith("Matplotlib"):
                fig = plot_prcc_bar_matplotlib(prcc_df, title=f"PRCC for {out_col}")
                st.pyplot(fig)
            else:
                fig = plot_prcc_bar_plotly(prcc_df, title=f"PRCC for {out_col}")
                st.plotly_chart(fig, use_container_width=True)
            st.session_state["_sa_prcc_last"] = {"out": out_col, "df": prcc_df}

with tabs[4]:
    st.subheader("Morris screening")
    if "Morris" not in methods:
        st.info("No Morris design found in this run.")
    else:
        ok_df = df[df["__method"] == "Morris"].copy()
        ok_df = ok_df[ok_df.get("__status", "ok") == "ok"].copy()
        out_col = st.selectbox("Output", options=outputs, key="morris_out")
        if st.button("Compute Morris indices", type="primary"):
            with st.spinner("Computing Morris indices..."):
                morris_df = compute_morris(df=df, design_meta=design_meta, output_col=out_col)
            st.dataframe(morris_df, use_container_width=True)
            if plot_backend.startswith("Matplotlib"):
                fig = plot_morris_scatter_matplotlib(morris_df, title=f"Morris μ* vs σ for {out_col}")
                st.pyplot(fig)
            else:
                fig = plot_morris_scatter_plotly(morris_df, title=f"Morris μ* vs σ for {out_col}")
                st.plotly_chart(fig, use_container_width=True)
            st.session_state["_sa_morris_last"] = {"out": out_col, "df": morris_df}

with tabs[5]:
    st.subheader("Sobol indices")
    if "Sobol" not in methods:
        st.info("No Sobol design found in this run.")
    else:
        out_col = st.selectbox("Output", options=outputs, key="sobol_out")
        if st.button("Compute Sobol indices", type="primary"):
            with st.spinner("Computing Sobol indices..."):
                sobol_df = compute_sobol(df=df, design_meta=design_meta, output_col=out_col)
            st.dataframe(sobol_df, use_container_width=True)
            if plot_backend.startswith("Matplotlib"):
                fig = plot_sobol_indices_matplotlib(sobol_df, title=f"Sobol indices for {out_col}")
                st.pyplot(fig)
            else:
                fig = plot_sobol_indices_plotly(sobol_df, title=f"Sobol indices for {out_col}")
                st.plotly_chart(fig, use_container_width=True)
            st.session_state["_sa_sobol_last"] = {"out": out_col, "df": sobol_df}

with tabs[6]:
    st.subheader("Download exports")
    # CSV download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="fgem_sensitivity_results.csv",
        mime="text/csv",
    )

    # ZIP bundle
    bundle_name = f"fgem_sensitivity_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

    def _make_bundle() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("results/results.csv", df.to_csv(index=False))
            zf.writestr("design/meta.json", json.dumps(design_meta, indent=2, default=str))
            # include last computed metric tables if present
            for key, fname in [
                ("_sa_prcc_last", "metrics/prcc.csv"),
                ("_sa_morris_last", "metrics/morris.csv"),
                ("_sa_sobol_last", "metrics/sobol.csv"),
            ]:
                v = st.session_state.get(key)
                if v and isinstance(v, dict) and "df" in v:
                    zf.writestr(fname, v["df"].to_csv(index=False))
        buf.seek(0)
        return buf.read()

    st.download_button(
        "Download analysis bundle (ZIP)",
        data=_make_bundle(),
        file_name=bundle_name,
        mime="application/zip",
        help="Includes results CSV + design metadata + any computed metric tables.",
    )

# Navigation
st.markdown("---")
nav1, nav2 = st.columns([1, 1])
with nav1:
    if st.button("◀️ Back to Sensitivity Analysis"):
        st.switch_page("pages/4_Sensitivity_Analysis.py")
with nav2:
    if st.button("◀️ Back to Results"):
        st.switch_page("pages/3_Results.py")

