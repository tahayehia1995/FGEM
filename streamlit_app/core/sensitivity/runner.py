"""Parallel batch runner for sensitivity analysis designs."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config_manager import ConfigManager


def _apply_overrides(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply overrides to a baseline config, supporting pseudo-keys.

    Reserved metadata keys are prefixed with '__' and ignored here.
    Supported pseudo-keys:
    - battery_duration_0 / battery_duration_1
    - battery_power_capacity_0 / battery_power_capacity_1
    - reservoir_simulator_settings.<subkey>
    """
    cfg = dict(base_config)

    # Ensure mutable structures exist
    if not isinstance(cfg.get("battery_duration"), list):
        cfg["battery_duration"] = list(cfg.get("battery_duration") or [0, 0])
    while len(cfg["battery_duration"]) < 2:
        cfg["battery_duration"].append(0.0)
    if not isinstance(cfg.get("battery_power_capacity"), list):
        cfg["battery_power_capacity"] = list(cfg.get("battery_power_capacity") or [0, 0])
    while len(cfg["battery_power_capacity"]) < 2:
        cfg["battery_power_capacity"].append(0.0)
    if not isinstance(cfg.get("reservoir_simulator_settings"), dict):
        cfg["reservoir_simulator_settings"] = dict(cfg.get("reservoir_simulator_settings") or {})

    for k, v in overrides.items():
        if k.startswith("__"):
            continue
        if k.startswith("reservoir_simulator_settings."):
            sub = k.split(".", 1)[1]
            cfg["reservoir_simulator_settings"][sub] = v
            continue
        if k == "battery_duration_0":
            cfg["battery_duration"][0] = float(v)
            continue
        if k == "battery_duration_1":
            cfg["battery_duration"][1] = float(v)
            continue
        if k == "battery_power_capacity_0":
            cfg["battery_power_capacity"][0] = float(v)
            continue
        if k == "battery_power_capacity_1":
            cfg["battery_power_capacity"][1] = float(v)
            continue

        if k == "resample" and (v is None or str(v).lower() in ("none", "false", "0", "")):
            cfg["resample"] = None
            continue

        cfg[k] = v

    return cfg


def _extract_outputs(world: Any, outputs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in outputs:
        if hasattr(world, name):
            out[name] = getattr(world, name)
        else:
            # common aliases
            aliases = {
                "NET_GEN": "NET_GEN",
                "NET_CF": "NET_CF",
                "AVG_T_AMB": "AVG_T_AMB",
                "PBP": "PBP",
            }
            attr = aliases.get(name)
            out[name] = getattr(world, attr, np.nan) if attr else np.nan
    return out


def _run_one(
    *,
    base_config: Dict[str, Any],
    config_manager: ConfigManager,
    case: Dict[str, Any],
    outputs: Sequence[str],
) -> Dict[str, Any]:
    t0 = time.time()
    meta = {k: v for k, v in case.items() if k.startswith("__")}
    overrides = {k: v for k, v in case.items() if not k.startswith("__")}
    row: Dict[str, Any] = {}
    row.update(meta)
    row.update(overrides)

    try:
        cfg0 = _apply_overrides(base_config, overrides)
        merged = config_manager.merge_config(cfg0, config_file_path=None)

        from ...fgem.world import World

        world = World(merged, reset_market_weather=True, config_manager=config_manager)
        for _ in range(world.max_simulation_steps):
            world.step_update_record()
        world.postprocess(print_outputs=False, compute_pumping=True)

        row.update(_extract_outputs(world, outputs))
        row["__status"] = "ok"
        row["__error"] = ""
    except Exception as e:
        row["__status"] = "error"
        row["__error"] = str(e)
        for name in outputs:
            row[name] = np.nan

    row["__runtime_s"] = float(time.time() - t0)
    return row


def run_batch(
    *,
    base_config: Dict[str, Any],
    config_manager: ConfigManager,
    cases: List[Dict[str, Any]],
    outputs: Sequence[str],
    n_jobs: int,
    chunk_size: int,
    on_update: Optional[Callable[[int, int, pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """Run a batch of sensitivity cases with chunked progress updates."""

    total = len(cases)
    rows: List[Dict[str, Any]] = []
    done = 0

    # Chunked parallel execution for UI responsiveness
    for start in range(0, total, max(1, int(chunk_size))):
        chunk = cases[start : start + int(chunk_size)]

        chunk_rows = Parallel(n_jobs=int(n_jobs), backend="loky")(
            delayed(_run_one)(
                base_config=base_config,
                config_manager=config_manager,
                case=case,
                outputs=outputs,
            )
            for case in chunk
        )
        rows.extend(chunk_rows)
        done += len(chunk_rows)

        df_partial = pd.DataFrame(rows)
        if on_update:
            on_update(done, total, df_partial)

    return pd.DataFrame(rows)

