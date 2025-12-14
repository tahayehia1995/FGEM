"""Sensitivity analysis setup + batch runner page for FGEM Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.core.config_manager import ConfigManager
from streamlit_app.core.defaults import get_defaults


def _init_sa_state():
    if "sa_run" not in st.session_state:
        st.session_state.sa_run = None
    if "sa_design" not in st.session_state:
        st.session_state.sa_design = None


st.set_page_config(page_title="Sensitivity - FGEM", page_icon="🧪", layout="wide")

st.title("🧪 Sensitivity Analysis")
st.markdown(
    "Configure and run sensitivity/uncertainty analyses (OAT, grid, Monte Carlo, PRCC, Morris, Sobol) "
    "across upstream/downstream/market/storage parameters."
)

_init_sa_state()

if "config" not in st.session_state:
    st.warning("⚠️ No configuration found. Please configure your project first.")
    if st.button("Go to Configuration"):
        st.switch_page("pages/1_Configuration.py")
    st.stop()

config_manager = ConfigManager()
base_config = st.session_state.config
defaults = get_defaults().to_dict()

st.header("📋 Baseline configuration")
with st.expander("View baseline configuration (used as the center point)", expanded=False):
    st.json(base_config)

# Lazy imports for sensitivity modules (avoid import cost on app load)
from streamlit_app.core.sensitivity.parameter_catalog import build_parameter_catalog
from streamlit_app.core.sensitivity.designs import (
    build_design,
    estimate_design_size,
    has_salib,
)
from streamlit_app.core.sensitivity.runner import run_batch

catalog = build_parameter_catalog(defaults=defaults, config_manager=config_manager)

st.header("🧰 Sensitivity design")

colA, colB, colC = st.columns([1.2, 1.0, 1.0])

with colA:
    methods = st.multiselect(
        "Methods",
        options=["OAT", "Grid/Factorial", "MonteCarlo", "PRCC", "Morris", "Sobol"],
        default=["MonteCarlo", "PRCC"],
        help="Select one or more methods. Some methods share the same Monte Carlo design (e.g., PRCC).",
    )
with colB:
    random_seed = st.number_input("Random seed", min_value=0, value=123, step=1)
with colC:
    n_jobs = st.number_input(
        "Parallel workers (n_jobs)",
        min_value=1,
        value=int(base_config.get("n_jobs", 1) or 1),
        step=1,
        help="Number of parallel simulations to run.",
    )

with st.expander("⚙️ Advanced design settings (sampling, levels, caps)", expanded=False):
    s1, s2, s3 = st.columns(3)
    with s1:
        mc_samples = st.number_input("Monte Carlo samples (MC/PRCC)", min_value=10, value=1000, step=50)
        oat_samples_per_param = st.number_input("OAT samples per parameter (for distributions)", min_value=3, value=11, step=2)
    with s2:
        grid_samples_per_param = st.number_input("Grid samples per parameter (for distributions)", min_value=2, value=5, step=1)
        grid_max_cases = st.number_input("Grid max cases (cap)", min_value=10, value=5000, step=100)
    with s3:
        morris_levels = st.number_input("Morris levels", min_value=2, value=4, step=1)
        morris_trajectories = st.number_input("Morris trajectories (N)", min_value=2, value=20, step=1)
        sobol_base_samples = st.number_input("Sobol base samples (N)", min_value=64, value=256, step=64)
        sobol_calc_second_order = st.checkbox("Sobol second-order indices (S2)", value=False)

method_settings = {
    "mc_samples": int(mc_samples),
    "oat_samples_per_param": int(oat_samples_per_param),
    "grid_samples_per_param": int(grid_samples_per_param),
    "grid_max_cases": int(grid_max_cases),
    "morris_levels": int(morris_levels),
    "morris_trajectories": int(morris_trajectories),
    "sobol_base_samples": int(sobol_base_samples),
    "sobol_calc_second_order": bool(sobol_calc_second_order),
}

st.subheader("🎯 Parameter selection")
param_keys = sorted(catalog.keys())

filter_cols = st.columns([1.0, 1.0, 1.0, 1.0])
with filter_cols[0]:
    search = st.text_input("Search parameters", value="")
with filter_cols[1]:
    categories = sorted({catalog[k]["category"] for k in param_keys})
    category_filter = st.multiselect("Category filter", options=categories, default=categories)
with filter_cols[2]:
    only_numeric = st.checkbox("Only numeric", value=False)
with filter_cols[3]:
    include_categorical = st.checkbox("Include categorical", value=True)

def _key_visible(k: str) -> bool:
    meta = catalog[k]
    if meta["category"] not in category_filter:
        return False
    if search and (search.lower() not in (k.lower() + " " + meta.get("label", "").lower())):
        return False
    if only_numeric and meta["kind"] not in ("float", "int"):
        return False
    if not include_categorical and meta["kind"] in ("categorical", "bool"):
        return False
    return True

visible_keys = [k for k in param_keys if _key_visible(k)]

selection_mode = st.radio(
    "Selection mode",
    options=["Single parameter", "Multiple parameters"],
    horizontal=True,
    help="Choose whether to sweep one parameter or run a multi-parameter design.",
)

if selection_mode == "Single parameter":
    selected = st.selectbox("Parameter", options=visible_keys)
    selected_params = [selected] if selected else []
else:
    selected_params = st.multiselect(
        "Parameters",
        options=visible_keys,
        default=[k for k in ["Tres_init", "well_tvd", "drilling_cost", "energy_price"] if k in visible_keys][:4],
    )

st.subheader("🧮 Per-parameter ranges / distributions")

param_specs: Dict[str, Dict[str, Any]] = {}

for k in selected_params:
    meta = catalog[k]
    with st.expander(f"{k} — {meta.get('label','')}".strip(" —"), expanded=True):
        st.caption(f"Category: **{meta['category']}** · Type: **{meta['kind']}** · Units: **{meta.get('units','')}**")

        if meta["kind"] in ("float", "int"):
            default_center = float(base_config.get(k, defaults.get(k, meta.get("default", 0.0)) or 0.0))
            suggested_min = meta.get("suggested_min")
            suggested_max = meta.get("suggested_max")
            if suggested_min is None or suggested_max is None:
                span = abs(default_center) if default_center != 0 else 1.0
                suggested_min = default_center - 0.25 * span
                suggested_max = default_center + 0.25 * span

            spec_type = st.selectbox(
                "Specification",
                options=["Range (min/max + count)", "Distribution"],
                index=0,
                key=f"spec_type_{k}",
            )

            if spec_type.startswith("Range"):
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                with c1:
                    vmin = st.number_input("Min", value=float(suggested_min), key=f"min_{k}")
                with c2:
                    vmax = st.number_input("Max", value=float(suggested_max), key=f"max_{k}")
                with c3:
                    count = st.number_input("Number of values", min_value=2, value=7, step=1, key=f"count_{k}")
                with c4:
                    spacing = st.selectbox("Spacing", options=["linear", "log"], index=0, key=f"spacing_{k}")
                param_specs[k] = {"mode": "range", "min": vmin, "max": vmax, "count": int(count), "spacing": spacing}
            else:
                dist = st.selectbox(
                    "Distribution",
                    options=["uniform", "loguniform", "normal", "triangular"],
                    index=0,
                    key=f"dist_{k}",
                )
                if dist in ("uniform", "loguniform"):
                    c1, c2 = st.columns(2)
                    with c1:
                        a = st.number_input("Low", value=float(suggested_min), key=f"low_{k}")
                    with c2:
                        b = st.number_input("High", value=float(suggested_max), key=f"high_{k}")
                    param_specs[k] = {"mode": "distribution", "dist": dist, "low": a, "high": b}
                elif dist == "normal":
                    c1, c2 = st.columns(2)
                    with c1:
                        mu = st.number_input("Mean (μ)", value=float(default_center), key=f"mu_{k}")
                    with c2:
                        sigma = st.number_input("Std dev (σ)", min_value=0.0, value=float(abs(default_center) * 0.1 + 1e-6), key=f"sigma_{k}")
                    param_specs[k] = {"mode": "distribution", "dist": dist, "mu": mu, "sigma": sigma}
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        left = st.number_input("Left", value=float(suggested_min), key=f"tri_left_{k}")
                    with c2:
                        mode = st.number_input("Mode", value=float(default_center), key=f"tri_mode_{k}")
                    with c3:
                        right = st.number_input("Right", value=float(suggested_max), key=f"tri_right_{k}")
                    param_specs[k] = {"mode": "distribution", "dist": dist, "left": left, "mode_value": mode, "right": right}

        elif meta["kind"] == "bool":
            choices = st.multiselect("Allowed values", options=[True, False], default=[True, False], key=f"bool_{k}")
            param_specs[k] = {"mode": "categorical", "values": list(choices)}
        else:
            values = meta.get("choices", [])
            default_vals = values[:1] if values else []
            selected_vals = st.multiselect("Allowed values", options=values, default=default_vals, key=f"cat_{k}")
            param_specs[k] = {"mode": "categorical", "values": list(selected_vals)}

st.subheader("📤 Outputs (metrics) to extract")
default_outputs = ["NPV", "LCOE", "IRR", "ROI", "PBP", "NET_GEN", "NET_CF"]
available_outputs = [
    "NPV", "LCOE", "IRR", "ROI", "PBP", "NET_GEN", "NET_CF", "AVG_T_AMB",
    "PPA_NPV", "PPA_IRR", "PPA_ROI", "PPA_PBP",
]
outputs = st.multiselect(
    "Select output metrics",
    options=available_outputs,
    default=[o for o in default_outputs if o in available_outputs],
)

st.subheader("🧾 Run size preview")
try:
    est = estimate_design_size(methods=methods, param_specs=param_specs, method_settings=method_settings)
    st.info(f"Estimated simulations: **{est:,}** (before failure filtering).")
except Exception as e:
    st.warning(f"Could not estimate run size: {e}")

st.header("🚀 Run sensitivity batch")

chunk_size = st.number_input(
    "Chunk size (cases per batch update)",
    min_value=1,
    value=25,
    step=1,
    help="Controls UI responsiveness and how often progress updates.",
)

col_run1, col_run2, col_run3 = st.columns([1, 1, 2])
with col_run1:
    build_only = st.checkbox("Build design only (do not run)", value=False)
with col_run2:
    show_cases = st.checkbox("Preview first 20 cases", value=True)
with col_run3:
    run_btn = st.button("▶️ Build + Run", type="primary")

if run_btn:
    # Friendly guard for optional dependency
    if any(m in methods for m in ["Morris", "Sobol"]) and not has_salib():
        st.error(
            "SALib is not installed in your current environment, but you selected Morris and/or Sobol.\n\n"
            "Fix options:\n"
            "- Deselect **Morris** and **Sobol**, or\n"
            "- Install SALib (recommended): `pip install SALib` (or reinstall requirements).\n"
        )
        st.stop()

    try:
        design = build_design(
            base_config=base_config,
            catalog=catalog,
            methods=methods,
            param_specs=param_specs,
            seed=int(random_seed),
            method_settings=method_settings,
        )
    except ModuleNotFoundError as e:
        st.error(str(e))
        st.stop()
    st.session_state.sa_design = design

    if show_cases:
        st.markdown("**Design preview (first 20 cases)**")
        st.dataframe(pd.DataFrame(design["cases"][:20]).fillna(""), use_container_width=True)

    if build_only:
        st.success("Design created. Switch to Sensitivity Results to analyze/visualize.")
        st.stop()

    with st.spinner("Running sensitivity batch..."):
        progress = st.progress(0)
        status = st.empty()
        table = st.empty()

        def _on_update(done: int, total: int, df_partial: pd.DataFrame):
            pct = 0 if total == 0 else int(100 * done / total)
            progress.progress(min(pct, 100))
            status.markdown(f"**Completed:** {done:,} / {total:,}  \n**Progress:** {pct}%")
            table.dataframe(df_partial.tail(200), use_container_width=True)

        df = run_batch(
            base_config=base_config,
            config_manager=config_manager,
            cases=design["cases"],
            outputs=outputs,
            n_jobs=int(n_jobs),
            chunk_size=int(chunk_size),
            on_update=_on_update,
        )

    st.session_state.sa_run = {
        "design": design,
        "results": df,
        "outputs": outputs,
        "base_config": base_config,
    }
    st.success("✅ Sensitivity batch completed. Open Sensitivity Results for plots + exports.")
    st.page_link("pages/5_Sensitivity_Results.py", label="Go to Sensitivity Results ▶️")

# Navigation
st.markdown("---")
nav1, nav2 = st.columns([1, 1])
with nav1:
    if st.button("◀️ Back to Results"):
        st.switch_page("pages/3_Results.py")
with nav2:
    if st.button("Next: Sensitivity Results ▶️"):
        st.switch_page("pages/5_Sensitivity_Results.py")

