from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mark10.algorithms import (
    QuantizedOracleResult,
    benefit_resource_repair,
    capacity_greedy,
    full_single_flip_check,
    lagrangian_relaxation,
    largest_memory_repair,
    legacy_count_br,
    quantized_oracle,
    random_feasible,
    run_wa_mcbr,
    quantized_feasible_projection,
    swap_improve,
)
from mark10.io_utils import ROOT, load_config, write_csv
from mark10.model import Scenario, generate_wireless_instance


PROFILE_DIR = ROOT / "02_任务池与画像"
RUN_DIR = ROOT / "03_逐运行结果"
TABLE_DIR = ROOT / "04_汇总表格"


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def wireless_seed(config: dict[str, Any], task_pool_id: int, wireless_index: int) -> int:
    return int(config["seed"]) + task_pool_id * 10000 + wireless_index * 97


def expected_main_instances(config: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (pool_id, index)
        for pool_id in range(1, int(config["n_task_pools"]) + 1)
        for index in range(int(config["wireless_seeds_per_pool"]))
    ]


def binding_capacity_pairs(config: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(work), float(memory))
        for work, memory in itertools.product(
            config["workload_capacity_fractions"], config["memory_available_grid_gb"]
        )
    ]


def binding_shard_paths(task_pool_id: int) -> tuple[Path, Path]:
    return (
        RUN_DIR / f"binding_resource_runs_pool{task_pool_id:02d}.csv",
        RUN_DIR / f"binding_oracle_diagnostics_pool{task_pool_id:02d}.csv",
    )


def profiling_error_conditions(config: dict[str, Any]) -> list[tuple[float, int, int, str]]:
    protections = ["no_margin", "fixed_margin", "conservative_quantization"]
    return [
        (float(magnitude), int(q_sign), int(v_sign), protection)
        for magnitude, (q_sign, v_sign), protection in itertools.product(
            config["profile_error_levels"],
            [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            protections,
        )
    ]


def load_task_profiles() -> pd.DataFrame:
    frame = pd.read_csv(PROFILE_DIR / "task_profiles.csv")
    if len(frame) != 500 or frame.task_uid.nunique() != 500:
        raise RuntimeError("Task profile file must contain 500 unique tasks")
    return frame


def _capacity_from_fraction(config: dict[str, Any], fraction: float) -> float:
    return (
        float(config["base_workload_capacity"])
        * fraction
        / (1.0 - float(config["workload_safety_epsilon"]))
    )


def _pool(tasks: pd.DataFrame, task_pool_id: int) -> pd.DataFrame:
    result = tasks.loc[tasks.task_pool_id.eq(task_pool_id)].sort_values("task_index").reset_index(drop=True)
    if len(result) != 100:
        raise RuntimeError(f"Task pool {task_pool_id} has {len(result)} tasks")
    return result


def build_scenario(
    tasks: pd.DataFrame,
    config: dict[str, Any],
    task_pool_id: int,
    wireless_index: int,
    *,
    workload_fraction: float,
    memory_available_gb: float,
    memory_mode: str = "hard",
    q_estimated: np.ndarray | None = None,
    v_estimated: np.ndarray | None = None,
    q_true: np.ndarray | None = None,
    v_true: np.ndarray | None = None,
    **radio_overrides: Any,
) -> Scenario:
    pool = _pool(tasks, task_pool_id)
    seed = wireless_seed(config, task_pool_id, wireless_index)
    wireless = generate_wireless_instance(pool, seed, config)
    q = wireless.q_llm.to_numpy(float) if q_estimated is None else np.asarray(q_estimated, dtype=float)
    return Scenario(
        wireless,
        q,
        config,
        v_estimated=v_estimated,
        q_true=q_true,
        v_true=v_true,
        workload_capacity=_capacity_from_fraction(config, workload_fraction),
        memory_available_gb=memory_available_gb,
        memory_mode=memory_mode,
        **radio_overrides,
    )


def _load_checkpoint(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    hashes = set(frame.config_hash.astype(str)) if "config_hash" in frame else set()
    expected = config_hash(config)
    if hashes != {expected}:
        raise RuntimeError(f"Checkpoint {path.name} does not match current configuration")
    return frame


def _checkpoint(path: Path, frame: pd.DataFrame) -> None:
    if not frame.empty:
        write_csv(path, frame)


def _oracle_diagnostics(
    oracle: QuantizedOracleResult,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config_hash": config_hash(config),
        **case,
        "quantized_objective": oracle.quantized_objective,
        "continuous_objective_of_quantized_strategy": oracle.continuous_objective,
        "k_offload": oracle.k_offload,
        "delta_q": float(config["oracle_delta_q"]),
        "delta_v_gb": float(config["oracle_delta_v_gb"]),
        "states_created": oracle.states_created,
        "states_pruned": oracle.states_pruned,
        "pruning_rate": oracle.pruning_rate,
        "peak_live_states": oracle.peak_live_states,
        "peak_python_memory_mb": oracle.peak_python_memory_mb,
        "runtime_seconds": oracle.runtime_seconds,
        "claim": oracle.claim,
    }


def _algorithm_record(
    scenario: Scenario,
    strategy: np.ndarray,
    algorithm: str,
    case: dict[str, Any],
    config: dict[str, Any],
    oracle: QuantizedOracleResult | None,
    *,
    runtime_seconds: float,
    updates: int = 0,
    changes: int = 0,
    swaps: int = 0,
    full_single_flip_pass: bool | None = None,
    signaling_payload_bytes: int = 0,
    signaling_header_bytes: int = 0,
    signaling_total_bytes: int = 0,
) -> dict[str, Any]:
    raw_quantized = scenario.quantized_objective(
        strategy, float(config["oracle_delta_q"]), float(config["oracle_delta_v_gb"])
    )
    projected = quantized_feasible_projection(
        scenario,
        strategy,
        float(config["oracle_delta_q"]),
        float(config["oracle_delta_v_gb"]),
    )
    projected_quantized = scenario.quantized_objective(
        projected, float(config["oracle_delta_q"]), float(config["oracle_delta_v_gb"])
    )
    if oracle is None:
        gap = math.nan
    else:
        gap = 100.0 * (projected_quantized - oracle.quantized_objective) / max(
            abs(oracle.quantized_objective), 1e-12
        )
    return {
        "config_hash": config_hash(config),
        **case,
        "algorithm": algorithm,
        "strategy_bits": "".join(str(int(value)) for value in strategy),
        "strategy_feasible_estimated": int(scenario.feasible(strategy)),
        "raw_strategy_quantized_feasible": int(math.isfinite(raw_quantized)),
        "raw_strategy_quantized_objective": raw_quantized,
        "quantized_projected_strategy_bits": "".join(str(int(value)) for value in projected),
        "quantized_projection_change_rate": float(np.mean(projected != strategy)),
        "projected_strategy_quantized_objective": projected_quantized,
        "quantized_oracle_objective": oracle.quantized_objective if oracle is not None else math.nan,
        "quantized_oracle_gap_percent": gap,
        "runtime_seconds": runtime_seconds,
        "updates": updates,
        "changes": changes,
        "swaps": swaps,
        "full_single_flip_pass": full_single_flip_pass,
        "signaling_payload_bytes": signaling_payload_bytes,
        "signaling_header_bytes": signaling_header_bytes,
        "signaling_total_bytes": signaling_total_bytes,
        **scenario.metrics(strategy),
    }


def _run_algorithms(
    scenario: Scenario,
    case: dict[str, Any],
    config: dict[str, Any],
    algorithm_seed: int,
    oracle: QuantizedOracleResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    strategies: list[dict[str, Any]] = []
    wa = run_wa_mcbr(scenario, algorithm_seed)
    records.append(
        _algorithm_record(
            scenario,
            wa.strategy,
            "wa_mcbr",
            case,
            config,
            oracle,
            runtime_seconds=wa.runtime_seconds,
            updates=wa.updates,
            changes=wa.changes,
            full_single_flip_pass=wa.full_single_flip_pass,
            signaling_payload_bytes=wa.signaling_payload_bytes,
            signaling_header_bytes=wa.signaling_header_bytes,
            signaling_total_bytes=wa.signaling_total_bytes,
        )
    )
    started = time.perf_counter()
    swapped, swaps = swap_improve(scenario, wa.strategy)
    swap_runtime = time.perf_counter() - started
    records.append(
        _algorithm_record(
            scenario,
            swapped,
            "wa_mcbr_swap",
            case,
            config,
            oracle,
            runtime_seconds=wa.runtime_seconds + swap_runtime,
            updates=wa.updates,
            changes=wa.changes,
            swaps=swaps,
            full_single_flip_pass=full_single_flip_check(
                scenario, swapped, float(config["epsilon_algorithm"])
            ),
            signaling_payload_bytes=wa.signaling_payload_bytes,
            signaling_header_bytes=wa.signaling_header_bytes,
            signaling_total_bytes=wa.signaling_total_bytes,
        )
    )
    started = time.perf_counter()
    greedy = capacity_greedy(scenario)
    records.append(
        _algorithm_record(
            scenario, greedy, "capacity_greedy", case, config, oracle,
            runtime_seconds=time.perf_counter() - started,
        )
    )
    lagrangian = lagrangian_relaxation(scenario)
    records.append(
        _algorithm_record(
            scenario,
            lagrangian.strategy,
            "lagrangian_relaxation",
            case,
            config,
            oracle,
            runtime_seconds=lagrangian.runtime_seconds,
            updates=lagrangian.updates,
            full_single_flip_pass=lagrangian.full_single_flip_pass,
        )
    )
    legacy = legacy_count_br(scenario, algorithm_seed)
    records.append(
        _algorithm_record(
            scenario,
            legacy.strategy,
            "legacy_count_br",
            case,
            config,
            oracle,
            runtime_seconds=legacy.runtime_seconds,
            updates=legacy.updates,
            changes=legacy.changes,
            full_single_flip_pass=legacy.full_single_flip_pass,
        )
    )
    started = time.perf_counter()
    random_strategy = random_feasible(scenario, algorithm_seed + 500000)
    records.append(
        _algorithm_record(
            scenario, random_strategy, "random_feasible", case, config, oracle,
            runtime_seconds=time.perf_counter() - started,
        )
    )
    all_local = np.zeros(scenario.n, dtype=np.int8)
    all_offload = np.ones(scenario.n, dtype=np.int8)
    records.append(_algorithm_record(scenario, all_local, "all_local", case, config, oracle, runtime_seconds=0.0))
    records.append(_algorithm_record(scenario, all_offload, "all_offload", case, config, oracle, runtime_seconds=0.0))
    records.append(
        _algorithm_record(
            scenario,
            oracle.strategy,
            "qdp_oracle",
            case,
            config,
            oracle,
            runtime_seconds=oracle.runtime_seconds,
        )
    )
    for record in records:
        strategies.append(
            {
                "config_hash": record["config_hash"],
                **case,
                "algorithm": record["algorithm"],
                "strategy_bits": record["strategy_bits"],
            }
        )
    return records, strategies


def _refresh_main_quantized_comparison(
    frame: pd.DataFrame,
    tasks: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    if "strategy_quantized_objective" in result:
        result = result.drop(columns=["strategy_quantized_objective"])
    scenario_cache: dict[tuple[int, int, str], Scenario] = {}
    columns: dict[str, list[Any]] = {
        "raw_strategy_quantized_feasible": [],
        "raw_strategy_quantized_objective": [],
        "quantized_projected_strategy_bits": [],
        "quantized_projection_change_rate": [],
        "projected_strategy_quantized_objective": [],
        "quantized_oracle_gap_percent": [],
    }
    delta_q = float(config["oracle_delta_q"])
    delta_v = float(config["oracle_delta_v_gb"])
    for row in result.itertuples(index=False):
        key = (int(row.task_pool_id), int(row.wireless_index), str(row.resource_scenario))
        if key not in scenario_cache:
            resources = config["resource_scenarios"][key[2]]
            scenario_cache[key] = build_scenario(
                tasks,
                config,
                key[0],
                key[1],
                workload_fraction=float(resources["workload_fraction"]),
                memory_available_gb=float(resources["memory_available_gb"]),
            )
        scenario = scenario_cache[key]
        strategy = np.array([int(value) for value in str(row.strategy_bits)], dtype=np.int8)
        raw = scenario.quantized_objective(strategy, delta_q, delta_v)
        projected = quantized_feasible_projection(scenario, strategy, delta_q, delta_v)
        projected_value = scenario.quantized_objective(projected, delta_q, delta_v)
        oracle_value = float(row.quantized_oracle_objective)
        columns["raw_strategy_quantized_feasible"].append(int(math.isfinite(raw)))
        columns["raw_strategy_quantized_objective"].append(raw)
        columns["quantized_projected_strategy_bits"].append(
            "".join(str(int(value)) for value in projected)
        )
        columns["quantized_projection_change_rate"].append(float(np.mean(projected != strategy)))
        columns["projected_strategy_quantized_objective"].append(projected_value)
        columns["quantized_oracle_gap_percent"].append(
            100.0 * (projected_value - oracle_value) / max(abs(oracle_value), 1e-12)
        )
    for name, values in columns.items():
        result[name] = values
    return result


def run_main_comparison(config: dict[str, Any] | None = None) -> None:
    config = load_config() if config is None else config
    tasks = load_task_profiles()
    run_path = RUN_DIR / "main_algorithm_runs.csv"
    oracle_path = RUN_DIR / "main_oracle_diagnostics.csv"
    strategy_path = RUN_DIR / "main_strategies.csv"
    trace_path = RUN_DIR / "representative_convergence_trace.csv"
    runs = _load_checkpoint(run_path, config)
    diagnostics = _load_checkpoint(oracle_path, config)
    strategies = _load_checkpoint(strategy_path, config)
    existing = set()
    if not runs.empty:
        counts = runs.groupby(["task_pool_id", "wireless_index", "resource_scenario"]).algorithm.nunique()
        existing = set(counts[counts >= 9].index)
    run_rows = runs.to_dict("records")
    diagnostic_rows = diagnostics.to_dict("records")
    strategy_rows = strategies.to_dict("records")
    total_cases = 5 * 10 * len(config["resource_scenarios"])
    completed = len(existing)
    for task_pool_id, wireless_index in expected_main_instances(config):
        for scenario_name, resources in config["resource_scenarios"].items():
            key = (task_pool_id, wireless_index, scenario_name)
            if key in existing:
                continue
            case = {
                "task_pool_id": task_pool_id,
                "wireless_index": wireless_index,
                "wireless_seed": wireless_seed(config, task_pool_id, wireless_index),
                "resource_scenario": scenario_name,
                "workload_capacity_fraction": float(resources["workload_fraction"]),
                "memory_available_gb_config": float(resources["memory_available_gb"]),
                "algorithm_seed": int(config["seed"]) + task_pool_id * 1000 + wireless_index,
            }
            scenario = build_scenario(
                tasks,
                config,
                task_pool_id,
                wireless_index,
                workload_fraction=float(resources["workload_fraction"]),
                memory_available_gb=float(resources["memory_available_gb"]),
            )
            oracle = quantized_oracle(
                scenario, float(config["oracle_delta_q"]), float(config["oracle_delta_v_gb"])
            )
            records, case_strategies = _run_algorithms(
                scenario, case, config, int(case["algorithm_seed"]), oracle
            )
            run_rows.extend(records)
            strategy_rows.extend(case_strategies)
            diagnostic_rows.append(_oracle_diagnostics(oracle, case, config))
            _checkpoint(run_path, pd.DataFrame(run_rows))
            _checkpoint(strategy_path, pd.DataFrame(strategy_rows))
            _checkpoint(oracle_path, pd.DataFrame(diagnostic_rows))
            completed += 1
            print(f"Main cases: {completed}/{total_cases}", flush=True)
    representative = build_scenario(
        tasks, config, 1, 0,
        workload_fraction=float(config["resource_scenarios"]["moderate"]["workload_fraction"]),
        memory_available_gb=float(config["resource_scenarios"]["moderate"]["memory_available_gb"]),
    )
    trace = run_wa_mcbr(representative, int(config["seed"]), record_trace=True)
    trace_rows = pd.DataFrame(trace.trace)
    trace_rows.insert(0, "config_hash", config_hash(config))
    write_csv(trace_path, trace_rows)
    refreshed = _refresh_main_quantized_comparison(pd.DataFrame(run_rows), tasks, config)
    write_csv(run_path, refreshed)


def _rejection_reasons(scenario: Scenario, strategy: np.ndarray) -> dict[str, int]:
    reasons = {"workload": 0, "memory": 0, "cost": 0}
    current = scenario.decision_objective(strategy)
    for index in np.flatnonzero(strategy == 0):
        candidate = strategy.copy()
        candidate[index] = 1
        _, workload, memory = scenario.state_values(candidate)
        if workload > scenario.workload_limit + 1e-12:
            reasons["workload"] += 1
        elif scenario.memory_mode == "hard" and memory > scenario.memory_available_gb + 1e-12:
            reasons["memory"] += 1
        elif scenario.decision_objective(candidate) >= current - float(scenario.config["epsilon_algorithm"]):
            reasons["cost"] += 1
    return reasons


def run_binding_grid(
    config: dict[str, Any] | None = None,
    task_pool_ids: list[int] | None = None,
) -> None:
    config = load_config() if config is None else config
    tasks = load_task_profiles()
    selected_pools = list(range(1, int(config["n_task_pools"]) + 1)) if task_pool_ids is None else task_pool_ids
    if task_pool_ids is not None and len(selected_pools) != 1:
        raise ValueError("A binding shard must contain exactly one task pool")
    if task_pool_ids is None:
        run_path = RUN_DIR / "binding_resource_runs.csv"
        oracle_path = RUN_DIR / "binding_oracle_diagnostics.csv"
    else:
        run_path, oracle_path = binding_shard_paths(selected_pools[0])
    runs = _load_checkpoint(run_path, config)
    diagnostics = _load_checkpoint(oracle_path, config)
    run_rows = runs.to_dict("records")
    diagnostic_rows = diagnostics.to_dict("records")
    existing = set()
    if not runs.empty:
        existing = set(
            zip(
                runs.task_pool_id,
                runs.wireless_index,
                runs.workload_capacity_fraction,
                runs.memory_available_gb_config,
                runs.memory_mode,
            )
        )
    instances = [item for item in expected_main_instances(config) if item[0] in selected_pools]
    total = len(instances) * len(binding_capacity_pairs(config)) * 3
    completed = len(existing)
    for task_pool_id, wireless_index in instances:
        for workload_fraction, memory_gb in binding_capacity_pairs(config):
            oracle: QuantizedOracleResult | None = None
            for memory_mode in ["hard", "soft", "none"]:
                key = (task_pool_id, wireless_index, workload_fraction, memory_gb, memory_mode)
                if key in existing:
                    continue
                case = {
                    "task_pool_id": task_pool_id,
                    "wireless_index": wireless_index,
                    "wireless_seed": wireless_seed(config, task_pool_id, wireless_index),
                    "workload_capacity_fraction": workload_fraction,
                    "memory_available_gb_config": memory_gb,
                    "memory_mode": memory_mode,
                }
                scenario = build_scenario(
                    tasks, config, task_pool_id, wireless_index,
                    workload_fraction=workload_fraction,
                    memory_available_gb=memory_gb,
                    memory_mode=memory_mode,
                )
                if memory_mode == "hard":
                    oracle = quantized_oracle(
                        scenario, float(config["oracle_delta_q"]), float(config["oracle_delta_v_gb"])
                    )
                    diagnostic_rows.append(_oracle_diagnostics(oracle, case, config))
                result = run_wa_mcbr(
                    scenario, int(config["seed"]) + task_pool_id * 1000 + wireless_index
                )
                record = _algorithm_record(
                    scenario,
                    result.strategy,
                    f"wa_mcbr_{memory_mode}",
                    case,
                    config,
                    oracle if memory_mode == "hard" else None,
                    runtime_seconds=result.runtime_seconds,
                    updates=result.updates,
                    changes=result.changes,
                    full_single_flip_pass=result.full_single_flip_pass,
                    signaling_payload_bytes=result.signaling_payload_bytes,
                    signaling_header_bytes=result.signaling_header_bytes,
                    signaling_total_bytes=result.signaling_total_bytes,
                )
                reasons = _rejection_reasons(scenario, result.strategy)
                record.update({f"rejections_{name}": value for name, value in reasons.items()})
                run_rows.append(record)
                _checkpoint(run_path, pd.DataFrame(run_rows))
                _checkpoint(oracle_path, pd.DataFrame(diagnostic_rows))
                completed += 1
                if completed % 25 == 0:
                    label = f" pool {selected_pools[0]}" if task_pool_ids is not None else ""
                    print(f"Binding{label} cases: {completed}/{total}", flush=True)


def combine_binding_shards(config: dict[str, Any] | None = None) -> None:
    config = load_config() if config is None else config
    base_run_path = RUN_DIR / "binding_resource_runs.csv"
    base_oracle_path = RUN_DIR / "binding_oracle_diagnostics.csv"
    run_frames = []
    oracle_frames = []
    if base_run_path.exists():
        base = pd.read_csv(base_run_path)
        run_frames.append(base.loc[base.task_pool_id.eq(1)])
    if base_oracle_path.exists():
        base = pd.read_csv(base_oracle_path)
        oracle_frames.append(base.loc[base.task_pool_id.eq(1)])
    for pool_id in range(2, int(config["n_task_pools"]) + 1):
        run_path, oracle_path = binding_shard_paths(pool_id)
        run_frames.append(pd.read_csv(run_path))
        oracle_frames.append(pd.read_csv(oracle_path))
    runs = pd.concat(run_frames, ignore_index=True).drop_duplicates(
        [
            "task_pool_id",
            "wireless_index",
            "workload_capacity_fraction",
            "memory_available_gb_config",
            "memory_mode",
        ],
        keep="last",
    )
    diagnostics = pd.concat(oracle_frames, ignore_index=True).drop_duplicates(
        [
            "task_pool_id",
            "wireless_index",
            "workload_capacity_fraction",
            "memory_available_gb_config",
        ],
        keep="last",
    )
    expected_runs = int(config["n_task_pools"]) * int(config["wireless_seeds_per_pool"]) * 25 * 3
    expected_oracles = int(config["n_task_pools"]) * int(config["wireless_seeds_per_pool"]) * 25
    if len(runs) != expected_runs or len(diagnostics) != expected_oracles:
        raise RuntimeError(
            f"Incomplete binding shards: runs={len(runs)}/{expected_runs}, "
            f"oracles={len(diagnostics)}/{expected_oracles}"
        )
    write_csv(base_run_path, runs.sort_values(
        ["task_pool_id", "wireless_index", "workload_capacity_fraction", "memory_available_gb_config", "memory_mode"]
    ))
    write_csv(base_oracle_path, diagnostics.sort_values(
        ["task_pool_id", "wireless_index", "workload_capacity_fraction", "memory_available_gb_config"]
    ))


def _jain_index(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    denominator = len(array) * float(np.square(array).sum())
    return float(array.sum() ** 2 / denominator) if denominator > 0 else math.nan


def _wireless_settings(config: dict[str, Any]) -> list[tuple[str, float, dict[str, float]]]:
    settings: list[tuple[str, float, dict[str, float]]] = []
    for value in config["bandwidth_sweep_mhz"]:
        settings.append(("bandwidth_mhz", float(value), {"bandwidth_hz": float(value) * 1e6}))
    for value in config["image_payload_sweep_mb"]:
        settings.append(
            (
                "image_payload_mb",
                float(value),
                {"input_image_payload_mb": float(value), "output_image_payload_mb": float(value)},
            )
        )
    for value in config["uplink_power_sweep_w"]:
        settings.append(("uplink_power_w", float(value), {"uplink_power_w": float(value)}))
    for value in config["downlink_power_sweep_w"]:
        settings.append(("downlink_power_w", float(value), {"downlink_power_w": float(value)}))
    for value in config["path_loss_exponent_sweep"]:
        settings.append(("path_loss_exponent", float(value), {"path_loss_exponent": float(value)}))
    return settings


def run_wireless_sensitivity(config: dict[str, Any] | None = None) -> None:
    config = load_config() if config is None else config
    tasks = load_task_profiles()
    run_rows = []
    layer_rows = []
    settings = _wireless_settings(config)
    total = 50 * len(settings)
    completed = 0
    moderate = config["resource_scenarios"]["moderate"]
    for task_pool_id, wireless_index in expected_main_instances(config):
        for factor, value, overrides in settings:
            scenario = build_scenario(
                tasks, config, task_pool_id, wireless_index,
                workload_fraction=float(moderate["workload_fraction"]),
                memory_available_gb=float(moderate["memory_available_gb"]),
                **overrides,
            )
            result = run_wa_mcbr(
                scenario, int(config["seed"]) + task_pool_id * 1000 + wireless_index
            )
            outcomes = scenario.per_task_outcomes(result.strategy)
            metrics = scenario.metrics(result.strategy)
            run_rows.append(
                {
                    "config_hash": config_hash(config),
                    "task_pool_id": task_pool_id,
                    "wireless_index": wireless_index,
                    "wireless_seed": wireless_seed(config, task_pool_id, wireless_index),
                    "factor": factor,
                    "value": value,
                    "jain_service_quality": _jain_index(outcomes.service_quality.to_numpy(float)),
                    "updates": result.updates,
                    "runtime_seconds": result.runtime_seconds,
                    **metrics,
                }
            )
            layers = pd.qcut(outcomes.distance_m, 3, labels=["near", "middle", "far"])
            for layer in ["near", "middle", "far"]:
                part = outcomes.loc[layers == layer]
                layer_rows.append(
                    {
                        "config_hash": config_hash(config),
                        "task_pool_id": task_pool_id,
                        "wireless_index": wireless_index,
                        "factor": factor,
                        "value": value,
                        "distance_layer": layer,
                        "n_users": len(part),
                        "mean_distance_m": float(part.distance_m.mean()),
                        "offload_rate": float(part.strategy.mean()),
                        "mean_end_to_end_delay_seconds": float(part.end_to_end_delay_seconds.mean()),
                        "mean_service_quality": float(part.service_quality.mean()),
                    }
                )
            completed += 1
            if completed % 50 == 0:
                print(f"Wireless cases: {completed}/{total}", flush=True)
    write_csv(RUN_DIR / "wireless_sensitivity_runs.csv", pd.DataFrame(run_rows))
    write_csv(RUN_DIR / "wireless_distance_layers.csv", pd.DataFrame(layer_rows))


def run_profile_error_robustness(config: dict[str, Any] | None = None) -> None:
    config = load_config() if config is None else config
    tasks = load_task_profiles()
    rows = []
    moderate = config["resource_scenarios"]["moderate"]
    conditions = profiling_error_conditions(config)
    total = 50 * len(conditions)
    completed = 0
    for task_pool_id, wireless_index in expected_main_instances(config):
        pool = _pool(tasks, task_pool_id)
        q_true = pool.q_llm.to_numpy(float)
        v_true = pool.vram_requirement_gb_simulated.to_numpy(float)
        truth_scenario = build_scenario(
            tasks, config, task_pool_id, wireless_index,
            workload_fraction=float(moderate["workload_fraction"]),
            memory_available_gb=float(moderate["memory_available_gb"]),
            q_estimated=q_true,
            v_estimated=v_true,
            q_true=q_true,
            v_true=v_true,
        )
        baseline = run_wa_mcbr(
            truth_scenario, int(config["seed"]) + task_pool_id * 1000 + wireless_index
        ).strategy
        baseline_objective = truth_scenario.public_objective(baseline)
        for magnitude, q_sign, v_sign, protection in conditions:
            q_estimated = q_true * (1.0 + q_sign * magnitude)
            v_estimated = v_true * (1.0 + v_sign * magnitude)
            if protection == "fixed_margin":
                margin = 1.0 + float(config["fixed_safety_margin"])
                q_estimated *= margin
                v_estimated *= margin
            elif protection == "conservative_quantization":
                dq = float(config["oracle_delta_q"])
                dv = float(config["oracle_delta_v_gb"])
                q_estimated = np.ceil(q_estimated / dq) * dq
                v_estimated = np.ceil(v_estimated / dv) * dv
            scenario = build_scenario(
                tasks, config, task_pool_id, wireless_index,
                workload_fraction=float(moderate["workload_fraction"]),
                memory_available_gb=float(moderate["memory_available_gb"]),
                q_estimated=q_estimated,
                v_estimated=v_estimated,
                q_true=q_true,
                v_true=v_true,
            )
            result = run_wa_mcbr(
                scenario, int(config["seed"]) + task_pool_id * 1000 + wireless_index
            )
            truth_value = truth_scenario.public_objective(result.strategy)
            rows.append(
                {
                    "config_hash": config_hash(config),
                    "task_pool_id": task_pool_id,
                    "wireless_index": wireless_index,
                    "error_magnitude": magnitude,
                    "q_error_sign": q_sign,
                    "v_error_sign": v_sign,
                    "protection": protection,
                    "strategy_change_rate": float(np.mean(result.strategy != baseline)),
                    "true_public_objective_J": truth_value,
                    "baseline_true_public_objective_J": baseline_objective,
                    "objective_degradation_percent": 100.0
                    * (truth_value - baseline_objective)
                    / max(abs(baseline_objective), 1e-12),
                    "updates": result.updates,
                    "runtime_seconds": result.runtime_seconds,
                    **scenario.metrics(result.strategy),
                }
            )
            completed += 1
            if completed % 100 == 0:
                print(f"Robustness cases: {completed}/{total}", flush=True)
    write_csv(RUN_DIR / "profiling_error_robustness.csv", pd.DataFrame(rows))


def run_scale_analysis(config: dict[str, Any] | None = None) -> None:
    config = load_config() if config is None else config
    tasks = load_task_profiles()
    combined = pd.concat([_pool(tasks, 1), _pool(tasks, 2)], ignore_index=True)
    run_rows = []
    oracle_rows = []
    for n in config["scale_n_values"]:
        subset = combined.iloc[: int(n)].copy().reset_index(drop=True)
        for wireless_index in range(int(config["wireless_seeds_per_pool"])):
            seed = wireless_seed(config, 90 + int(n), wireless_index)
            wireless = generate_wireless_instance(subset, seed, config)
            scenario = Scenario(wireless, wireless.q_llm.to_numpy(float), config)
            wa = run_wa_mcbr(scenario, int(config["seed"]) + int(n) * 100 + wireless_index)
            run_rows.append(
                {
                    "config_hash": config_hash(config),
                    "n_nodes": int(n),
                    "wireless_index": wireless_index,
                    "wireless_seed": seed,
                    "algorithm": "wa_mcbr",
                    "updates": wa.updates,
                    "changes": wa.changes,
                    "runtime_seconds": wa.runtime_seconds,
                    "signaling_total_bytes": wa.signaling_total_bytes,
                    "full_single_flip_pass": wa.full_single_flip_pass,
                    **scenario.metrics(wa.strategy),
                }
            )
            legacy = legacy_count_br(scenario, int(config["seed"]) + int(n) * 100 + wireless_index)
            run_rows.append(
                {
                    "config_hash": config_hash(config),
                    "n_nodes": int(n),
                    "wireless_index": wireless_index,
                    "wireless_seed": seed,
                    "algorithm": "legacy_count_br",
                    "updates": legacy.updates,
                    "changes": legacy.changes,
                    "runtime_seconds": legacy.runtime_seconds,
                    "signaling_total_bytes": 0,
                    "full_single_flip_pass": legacy.full_single_flip_pass,
                    **scenario.metrics(legacy.strategy),
                }
            )
            if wireless_index == 0:
                oracle = quantized_oracle(
                    scenario, float(config["oracle_delta_q"]), float(config["oracle_delta_v_gb"])
                )
                oracle_rows.append(
                    _oracle_diagnostics(
                        oracle,
                        {"n_nodes": int(n), "wireless_index": wireless_index, "wireless_seed": seed},
                        config,
                    )
                )
        print(f"Scale N={n} complete", flush=True)
    write_csv(RUN_DIR / "scale_runs.csv", pd.DataFrame(run_rows))
    write_csv(RUN_DIR / "scale_oracle_diagnostics.csv", pd.DataFrame(oracle_rows))


def run_failure_case_audit(config: dict[str, Any] | None = None) -> None:
    config = load_config() if config is None else config
    rows = []
    main = pd.read_csv(RUN_DIR / "main_algorithm_runs.csv")
    wa = main.loc[main.algorithm.eq("wa_mcbr")].copy()
    largest_gap = wa.replace([np.inf, -np.inf], np.nan).dropna(subset=["quantized_oracle_gap_percent"]).nlargest(
        1, "quantized_oracle_gap_percent"
    )
    for record in largest_gap.to_dict("records"):
        rows.append({"failure_region": "largest_single_flip_to_quantized_oracle_gap", **record})
    binding = pd.read_csv(RUN_DIR / "binding_resource_runs.csv")
    extreme = binding.loc[
        binding.workload_capacity_fraction.eq(min(config["workload_capacity_fractions"]))
        & binding.memory_available_gb_config.eq(min(config["memory_available_grid_gb"]))
    ]
    for record in extreme.groupby("memory_mode", as_index=False).first().to_dict("records"):
        rows.append({"failure_region": "extreme_joint_resource_constraint", **record})
    robustness = pd.read_csv(RUN_DIR / "profiling_error_robustness.csv")
    severe = robustness.loc[robustness.error_magnitude.eq(max(config["profile_error_levels"]))]
    for record in severe.nlargest(3, "objective_degradation_percent").to_dict("records"):
        rows.append({"failure_region": "severe_profile_error", **record})
    distance = pd.read_csv(RUN_DIR / "wireless_distance_layers.csv")
    far = distance.loc[distance.distance_layer.eq("far")]
    for record in far.nsmallest(3, "offload_rate").to_dict("records"):
        rows.append({"failure_region": "far_user_offload_suppression", **record})
    write_csv(RUN_DIR / "failure_cases.csv", pd.DataFrame(rows))


STAGES = {
    "main": run_main_comparison,
    "binding": run_binding_grid,
    "binding-combine": combine_binding_shards,
    "wireless": run_wireless_sensitivity,
    "robustness": run_profile_error_robustness,
    "scale": run_scale_analysis,
    "failures": run_failure_case_audit,
}
