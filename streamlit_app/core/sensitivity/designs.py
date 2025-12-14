"""Sensitivity design generation utilities (OAT, grid, Monte Carlo, Morris, Sobol)."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def has_salib() -> bool:
    try:
        import SALib  # noqa: F401
        return True
    except Exception:
        return False


def _linspace(min_v: float, max_v: float, n: int, spacing: str) -> np.ndarray:
    if n < 2:
        return np.array([min_v], dtype=float)
    if spacing == "log":
        a = float(min_v)
        b = float(max_v)
        # protect against invalid log bounds
        a = a if a > 0 else 1e-9
        b = b if b > 0 else 1e-9
        lo = np.log10(min(a, b))
        hi = np.log10(max(a, b))
        vals = np.power(10.0, np.linspace(lo, hi, n))
        return vals
    return np.linspace(float(min_v), float(max_v), n)


def _sample_distribution(spec: Dict[str, Any], n: int, rng: np.random.Generator) -> np.ndarray:
    dist = spec["dist"]
    if dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size=n)
    if dist == "loguniform":
        lo = float(spec["low"])
        hi = float(spec["high"])
        lo = lo if lo > 0 else 1e-9
        hi = hi if hi > 0 else 1e-9
        return np.exp(rng.uniform(np.log(min(lo, hi)), np.log(max(lo, hi)), size=n))
    if dist == "normal":
        return rng.normal(float(spec["mu"]), float(spec["sigma"]), size=n)
    if dist == "triangular":
        left = float(spec["left"])
        mode = float(spec["mode_value"])
        right = float(spec["right"])
        return rng.triangular(left, mode, right, size=n)
    raise ValueError(f"Unknown distribution: {dist}")

def _bounds_from_spec(spec: Dict[str, Any]) -> Tuple[float, float]:
    if spec.get("mode") == "range":
        a = float(spec["min"])
        b = float(spec["max"])
        return (min(a, b), max(a, b))
    if spec.get("mode") == "distribution":
        dist = spec["dist"]
        if dist in ("uniform", "loguniform"):
            a = float(spec["low"])
            b = float(spec["high"])
            return (min(a, b), max(a, b))
        if dist == "triangular":
            a = float(spec["left"])
            b = float(spec["right"])
            return (min(a, b), max(a, b))
        if dist == "normal":
            mu = float(spec["mu"])
            sigma = float(spec["sigma"])
            return (mu - 3.0 * sigma, mu + 3.0 * sigma)
    # Fallback (should not be used for SALib problems)
    return (0.0, 1.0)


def estimate_design_size(*, methods: Sequence[str], param_specs: Dict[str, Dict[str, Any]], method_settings: Optional[Dict[str, Any]] = None) -> int:
    """Rough design size estimate for UI preview."""
    s = method_settings or {}
    total = 0

    if "OAT" in methods:
        for spec in param_specs.values():
            if spec.get("mode") == "range":
                total += int(spec.get("count", 7))
            elif spec.get("mode") == "categorical":
                total += len(spec.get("values", []))
            else:
                total += int(s.get("oat_samples_per_param", 11))

    if "Grid/Factorial" in methods:
        sizes = []
        for spec in param_specs.values():
            if spec.get("mode") == "range":
                sizes.append(int(spec.get("count", 5)))
            elif spec.get("mode") == "categorical":
                sizes.append(max(1, len(spec.get("values", []))))
            else:
                sizes.append(int(s.get("grid_samples_per_param", 5)))
        grid = int(np.prod(sizes)) if sizes else 0
        cap = int(s.get("grid_max_cases", 5000))
        total += min(grid, cap)

    if "MonteCarlo" in methods or "PRCC" in methods:
        total += int(s.get("mc_samples", 1000))

    if "Morris" in methods:
        # SALib morris: number of model evaluations = N * (D + 1)
        N = int(s.get("morris_trajectories", 20))
        D = len(param_specs)
        total += max(0, N * (D + 1))

    if "Sobol" in methods:
        # Saltelli: N*(2D+2) (no second order)
        N = int(s.get("sobol_base_samples", 256))
        D = len(param_specs)
        total += max(0, N * (2 * D + 2))

    return int(total)


def _numeric_param_names(catalog: Dict[str, Dict[str, Any]], param_specs: Dict[str, Dict[str, Any]]) -> List[str]:
    names = []
    for k in param_specs.keys():
        kind = catalog.get(k, {}).get("kind")
        if kind in ("float", "int"):
            names.append(k)
    return names


def build_design(
    *,
    base_config: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    methods: Sequence[str],
    param_specs: Dict[str, Dict[str, Any]],
    seed: int,
    method_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a unified design containing cases tagged by method.

    Each case is a dict with:
    - overrides for FGEM config keys / pseudo-keys
    - reserved metadata keys prefixed with '__'
    """

    settings = {
        "mc_samples": 1000,
        "oat_samples_per_param": 11,
        "grid_max_cases": 5000,
        "grid_samples_per_param": 5,
        "morris_levels": 4,
        "morris_trajectories": 20,
        "sobol_base_samples": 256,
        "sobol_calc_second_order": False,
    }
    if method_settings:
        settings.update(method_settings)

    rng = np.random.default_rng(int(seed))

    cases: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"settings": settings, "methods": list(methods), "params": list(param_specs.keys())}

    def _append_case(method: str, override: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        c = {"__method": method}
        if extra:
            c.update(extra)
        c.update(override)
        cases.append(c)

    # OAT
    if "OAT" in methods:
        for p, spec in param_specs.items():
            if spec["mode"] == "range":
                vals = _linspace(spec["min"], spec["max"], int(spec["count"]), spec.get("spacing", "linear"))
            elif spec["mode"] == "categorical":
                vals = np.array(spec.get("values", []), dtype=object)
            else:
                vals = _sample_distribution(spec, int(settings["oat_samples_per_param"]), rng)
            for j, v in enumerate(vals):
                _append_case("OAT", {p: v}, extra={"__param": p, "__idx": int(j)})

    # Grid / factorial
    if "Grid/Factorial" in methods:
        axes: List[Tuple[str, List[Any]]] = []
        for p, spec in param_specs.items():
            if spec["mode"] == "range":
                vals = _linspace(spec["min"], spec["max"], int(spec["count"]), spec.get("spacing", "linear")).tolist()
            elif spec["mode"] == "categorical":
                vals = list(spec.get("values", []))
            else:
                vals = _sample_distribution(spec, int(settings["grid_samples_per_param"]), rng).tolist()
            axes.append((p, vals))
        if axes:
            grid = list(product(*[vals for _, vals in axes]))
            max_cases = int(settings["grid_max_cases"])
            if len(grid) > max_cases:
                idx = rng.choice(len(grid), size=max_cases, replace=False)
                grid = [grid[i] for i in idx]
            for gi, tup in enumerate(grid):
                _append_case(
                    "Grid/Factorial",
                    {axes[i][0]: tup[i] for i in range(len(axes))},
                    extra={"__idx": int(gi)},
                )

    # Monte Carlo (also used by PRCC)
    if ("MonteCarlo" in methods) or ("PRCC" in methods):
        N = int(settings["mc_samples"])
        for i in range(N):
            row: Dict[str, Any] = {}
            for p, spec in param_specs.items():
                if spec["mode"] == "range":
                    row[p] = float(rng.uniform(spec["min"], spec["max"]))
                elif spec["mode"] == "categorical":
                    vals = spec.get("values", [])
                    row[p] = vals[int(rng.integers(0, len(vals)))] if vals else None
                else:
                    row[p] = float(_sample_distribution(spec, 1, rng)[0])
            _append_case("MonteCarlo", row, extra={"__idx": int(i)})
        if "PRCC" in methods:
            meta["prcc_source_method"] = "MonteCarlo"

    # Morris and Sobol require numeric parameters
    numeric_params = _numeric_param_names(catalog, param_specs)
    numeric_specs = {k: param_specs[k] for k in numeric_params}

    if "Morris" in methods and numeric_params:
        if not has_salib():
            raise ModuleNotFoundError(
                "SALib is required for Morris sensitivity. Install it (e.g., `pip install SALib`) "
                "or deselect the Morris method."
            )
        from SALib.sample.morris import sample as morris_sample

        problem = {
            "num_vars": len(numeric_params),
            "names": numeric_params,
            "bounds": [list(_bounds_from_spec(numeric_specs[p])) for p in numeric_params],
        }
        X = morris_sample(
            problem,
            N=int(settings["morris_trajectories"]),
            num_levels=int(settings["morris_levels"]),
            optimal_trajectories=None,
            seed=int(seed),
        )
        meta["morris_problem"] = problem
        meta["morris_X_shape"] = list(X.shape)
        for i in range(X.shape[0]):
            _append_case("Morris", {numeric_params[j]: float(X[i, j]) for j in range(X.shape[1])}, extra={"__sample_idx": int(i)})

    if "Sobol" in methods and numeric_params:
        if not has_salib():
            raise ModuleNotFoundError(
                "SALib is required for Sobol sensitivity. Install it (e.g., `pip install SALib`) "
                "or deselect the Sobol method."
            )
        from SALib.sample.sobol import sample as sobol_sample

        problem = {
            "num_vars": len(numeric_params),
            "names": numeric_params,
            "bounds": [list(_bounds_from_spec(numeric_specs[p])) for p in numeric_params],
        }
        X = sobol_sample(
            problem,
            N=int(settings["sobol_base_samples"]),
            calc_second_order=bool(settings["sobol_calc_second_order"]),
            scramble=True,
            seed=int(seed),
        )
        meta["sobol_problem"] = problem
        meta["sobol_X_shape"] = list(X.shape)
        meta["sobol_calc_second_order"] = bool(settings["sobol_calc_second_order"])
        for i in range(X.shape[0]):
            _append_case("Sobol", {numeric_params[j]: float(X[i, j]) for j in range(X.shape[1])}, extra={"__sample_idx": int(i)})

    meta["numeric_params_for_salib"] = numeric_params
    meta["total_cases"] = len(cases)

    return {"cases": cases, "meta": meta}

