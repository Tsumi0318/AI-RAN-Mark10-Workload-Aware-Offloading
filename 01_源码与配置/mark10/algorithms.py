from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import numpy as np

from mark10.model import Scenario


@dataclass
class GameResult:
    strategy: np.ndarray
    trace: list[dict[str, Any]]
    updates: int
    changes: int
    full_single_flip_pass: bool
    runtime_seconds: float
    signaling_payload_bytes: int = 0
    signaling_header_bytes: int = 0
    signaling_total_bytes: int = 0


@dataclass
class QuantizedOracleResult:
    strategy: np.ndarray
    quantized_objective: float
    continuous_objective: float
    q_integer: np.ndarray
    v_integer: np.ndarray
    k_offload: int
    states_created: int
    states_pruned: int
    pruning_rate: float
    peak_live_states: int
    peak_python_memory_mb: float
    runtime_seconds: float
    claim: str = "quantized_global_optimum_not_continuous_global_optimum"


def signaling_bytes_per_update(config: dict[str, Any]) -> dict[str, int]:
    sizes = config["signaling_bytes"]
    payload = (
        int(sizes["aggregate_broadcast_payload"])
        + int(sizes["selected_task_identifier"])
        + int(sizes["decision_reply_payload"])
        + int(sizes["update_ack_payload"])
    )
    headers = 4 * int(sizes["protocol_header_per_message"])
    return {"payload_bytes": payload, "header_bytes": headers, "total_bytes": payload + headers}


def full_single_flip_check(scenario: Scenario, strategy: np.ndarray, epsilon: float) -> bool:
    current = scenario.decision_objective(strategy)
    for index in range(scenario.n):
        candidate = strategy.copy()
        candidate[index] = 1 - candidate[index]
        if scenario.decision_objective(candidate) < current - epsilon:
            return False
    return True


def benefit_resource_repair(scenario: Scenario, strategy: np.ndarray) -> np.ndarray:
    result = np.asarray(strategy, dtype=np.int8).copy()
    while not scenario.feasible(result):
        selected = np.flatnonzero(result)
        if len(selected) == 0:
            raise AssertionError("Empty strategy should always satisfy positive capacities")
        k = max(int(result.sum()), 1)
        benefit = scenario.local_cost[selected] - scenario.offload_base[k, selected]
        burden = (
            scenario.q_estimated[selected] / max(scenario.workload_limit, 1e-12)
            + scenario.v_estimated[selected] / max(scenario.memory_available_gb, 1e-12)
        )
        score = benefit / np.maximum(burden, 1e-12)
        result[selected[int(np.argmin(score))]] = 0
    return result


def largest_memory_repair(scenario: Scenario, strategy: np.ndarray) -> np.ndarray:
    result = np.asarray(strategy, dtype=np.int8).copy()
    while not scenario.feasible(result):
        selected = np.flatnonzero(result)
        result[selected[int(np.argmax(scenario.v_estimated[selected]))]] = 0
    return result


def quantized_feasible_projection(
    scenario: Scenario,
    strategy: np.ndarray,
    delta_q: float,
    delta_v: float,
) -> np.ndarray:
    result = np.asarray(strategy, dtype=np.int8).copy()
    q_quantized = np.maximum(np.ceil(scenario.q_estimated / delta_q) * delta_q, delta_q)
    v_quantized = np.maximum(np.ceil(scenario.v_estimated / delta_v) * delta_v, delta_v)
    while True:
        selected = np.flatnonzero(result)
        workload = float(q_quantized[selected].sum())
        memory = float(v_quantized[selected].sum())
        if workload <= scenario.workload_limit + 1e-12 and memory <= scenario.memory_available_gb + 1e-12:
            return result
        k = max(len(selected), 1)
        benefit = scenario.local_cost[selected] - scenario.offload_base[k, selected]
        burden = (
            q_quantized[selected] / max(scenario.workload_limit, 1e-12)
            + v_quantized[selected] / max(scenario.memory_available_gb, 1e-12)
        )
        result[selected[int(np.argmin(benefit / np.maximum(burden, 1e-12)))]] = 0


def run_wa_mcbr(
    scenario: Scenario,
    seed: int,
    record_trace: bool = False,
    initial_strategy: np.ndarray | None = None,
) -> GameResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, scenario.n, dtype=np.int8) if initial_strategy is None else initial_strategy
    strategy = benefit_resource_repair(scenario, raw)
    epsilon = float(scenario.config["epsilon_algorithm"])
    trace: list[dict[str, Any]] = []
    idle = 0
    changes = 0
    updates = 0
    for iteration in range(int(scenario.config["max_updates"])):
        if idle >= scenario.n:
            if full_single_flip_check(scenario, strategy, epsilon):
                break
            idle = 0
        index = int(rng.integers(0, scenario.n))
        before = scenario.decision_objective(strategy)
        local = strategy.copy()
        local[index] = 0
        offload = strategy.copy()
        offload[index] = 1
        local_value = scenario.decision_objective(local)
        offload_value = scenario.decision_objective(offload)
        old = int(strategy[index])
        if offload_value < local_value - epsilon:
            strategy[index] = 1
        elif local_value < offload_value - epsilon:
            strategy[index] = 0
        changed = int(strategy[index]) != old
        after = scenario.decision_objective(strategy)
        if changed and not after < before - epsilon:
            raise AssertionError("Accepted WA-MCBR update did not strictly reduce the decision objective")
        if not scenario.feasible(strategy):
            raise AssertionError("WA-MCBR accepted an infeasible strategy")
        changes += int(changed)
        updates = iteration + 1
        idle = 0 if changed else idle + 1
        if record_trace:
            trace.append(
                {
                    "iteration": iteration,
                    "selected_task_index": index,
                    "strategy_before": old,
                    "strategy_after": int(strategy[index]),
                    "candidate_local_objective": local_value,
                    "candidate_offload_objective": offload_value,
                    "objective_before": before,
                    "objective_after": after,
                    "accepted": bool(changed),
                    **scenario.metrics(strategy),
                }
            )
    passed = full_single_flip_check(scenario, strategy, epsilon)
    bytes_per = signaling_bytes_per_update(scenario.config)
    return GameResult(
        strategy=strategy,
        trace=trace,
        updates=updates,
        changes=changes,
        full_single_flip_pass=passed,
        runtime_seconds=time.perf_counter() - started,
        signaling_payload_bytes=updates * bytes_per["payload_bytes"],
        signaling_header_bytes=updates * bytes_per["header_bytes"],
        signaling_total_bytes=updates * bytes_per["total_bytes"],
    )


def swap_improve(scenario: Scenario, strategy: np.ndarray) -> tuple[np.ndarray, int]:
    result = strategy.copy()
    epsilon = float(scenario.config["epsilon_algorithm"])
    swaps = 0
    while True:
        best_value = scenario.decision_objective(result)
        best_pair: tuple[int, int] | None = None
        for off in np.flatnonzero(result == 1):
            for local in np.flatnonzero(result == 0):
                candidate = result.copy()
                candidate[off] = 0
                candidate[local] = 1
                value = scenario.decision_objective(candidate)
                if value < best_value - epsilon:
                    best_value = value
                    best_pair = (int(off), int(local))
        if best_pair is None:
            return result, swaps
        result[best_pair[0]] = 0
        result[best_pair[1]] = 1
        swaps += 1


def capacity_greedy(scenario: Scenario) -> np.ndarray:
    strategy = np.zeros(scenario.n, dtype=np.int8)
    while True:
        current = scenario.decision_objective(strategy)
        best_value = current
        best_index: int | None = None
        for index in np.flatnonzero(strategy == 0):
            candidate = strategy.copy()
            candidate[index] = 1
            value = scenario.decision_objective(candidate)
            if value < best_value - float(scenario.config["epsilon_algorithm"]):
                best_value = value
                best_index = int(index)
        if best_index is None:
            return strategy
        strategy[best_index] = 1


def lagrangian_relaxation(scenario: Scenario) -> GameResult:
    started = time.perf_counter()
    lambda_work = 0.0
    lambda_memory = 0.0
    best = np.zeros(scenario.n, dtype=np.int8)
    best_value = scenario.decision_objective(best)
    iterations = int(scenario.config["lagrangian_iterations"])
    for iteration in range(iterations):
        standalone_benefit = scenario.local_cost - scenario.offload_base[1]
        penalized = (
            standalone_benefit
            - lambda_work * scenario.q_estimated
            - lambda_memory * scenario.v_estimated
        )
        raw = (penalized > 0).astype(np.int8)
        candidate = benefit_resource_repair(scenario, raw)
        value = scenario.decision_objective(candidate)
        if value < best_value:
            best = candidate
            best_value = value
        _, raw_workload, raw_memory = scenario.state_values(raw)
        step = float(scenario.config["lagrangian_step_initial"]) / math.sqrt(iteration + 1.0)
        lambda_work = max(0.0, lambda_work + step * (raw_workload - scenario.workload_limit))
        lambda_memory = max(0.0, lambda_memory + step * (raw_memory - scenario.memory_available_gb))
    return GameResult(
        strategy=best,
        trace=[],
        updates=iterations,
        changes=0,
        full_single_flip_pass=full_single_flip_check(
            scenario, best, float(scenario.config["epsilon_algorithm"])
        ),
        runtime_seconds=time.perf_counter() - started,
    )


def legacy_count_br(scenario: Scenario, seed: int) -> GameResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    strategy = benefit_resource_repair(scenario, rng.integers(0, 2, scenario.n, dtype=np.int8))
    idle = 0
    changes = 0
    updates = 0
    for iteration in range(int(scenario.config["max_updates"])):
        if idle >= scenario.n:
            break
        index = int(rng.integers(0, scenario.n))
        without = strategy.copy()
        without[index] = 0
        candidate = without.copy()
        candidate[index] = 1
        k = int(candidate.sum())
        off_cost = scenario.offload_base[k, index] + scenario.queue_delay(float(k))
        target = int(scenario.feasible(candidate) and off_cost < scenario.local_cost[index])
        old = int(strategy[index])
        strategy[index] = target
        changed = old != target
        changes += int(changed)
        idle = 0 if changed else idle + 1
        updates = iteration + 1
    return GameResult(
        strategy,
        [],
        updates,
        changes,
        full_single_flip_check(scenario, strategy, float(scenario.config["epsilon_algorithm"])),
        time.perf_counter() - started,
    )


def random_feasible(scenario: Scenario, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return benefit_resource_repair(scenario, rng.integers(0, 2, scenario.n, dtype=np.int8))


def _fenwick_query(tree: list[float], index: int) -> float:
    value = math.inf
    index += 1
    while index > 0:
        value = min(value, tree[index])
        index -= index & -index
    return value


def _fenwick_update(tree: list[float], index: int, value: float) -> None:
    index += 1
    while index < len(tree):
        tree[index] = min(tree[index], value)
        index += index & -index


def _dominance_prune(
    states: dict[tuple[int, int], tuple[float, int]], memory_cap: int
) -> tuple[dict[tuple[int, int], tuple[float, int]], int]:
    if len(states) <= 1:
        return states, 0
    ordered = sorted(
        ((work, memory, value[0], value[1]) for (work, memory), value in states.items()),
        key=lambda row: (row[0], row[1], row[2]),
    )
    tree = [math.inf] * (memory_cap + 2)
    kept: dict[tuple[int, int], tuple[float, int]] = {}
    for work, memory, cost, mask in ordered:
        if _fenwick_query(tree, memory) <= cost + 1e-12:
            continue
        kept[(work, memory)] = (cost, mask)
        _fenwick_update(tree, memory, cost)
    return kept, len(states) - len(kept)


def quantized_oracle(scenario: Scenario, delta_q: float, delta_v: float) -> QuantizedOracleResult:
    tracing_started = not tracemalloc.is_tracing()
    if tracing_started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    q_integer = np.maximum(np.ceil(scenario.q_estimated / delta_q).astype(int), 1)
    v_integer = np.maximum(np.ceil(scenario.v_estimated / delta_v).astype(int), 1)
    work_cap = int(math.floor(scenario.workload_limit / delta_q + 1e-12))
    memory_cap = int(math.floor(scenario.memory_available_gb / delta_v + 1e-12))
    all_local = float(scenario.local_cost.sum())
    best_value = all_local
    best_strategy = np.zeros(scenario.n, dtype=np.int8)
    states_created = 1
    states_pruned = 0
    peak_states = 1
    q_prefix = np.concatenate([[0], np.cumsum(np.sort(q_integer))])
    v_prefix = np.concatenate([[0], np.cumsum(np.sort(v_integer))])
    max_k = 0
    for k in range(1, scenario.n + 1):
        if q_prefix[k] <= work_cap and v_prefix[k] <= memory_cap:
            max_k = k
        else:
            break
    for target_k in range(1, max_k + 1):
        deltas = scenario.offload_base[target_k] - scenario.local_cost
        states: list[dict[tuple[int, int], tuple[float, int]]] = [dict() for _ in range(target_k + 1)]
        states[0][(0, 0)] = (0.0, 0)
        for index in range(scenario.n):
            remaining = scenario.n - index - 1
            next_states = [dict(group) for group in states]
            for k in range(min(target_k - 1, index), -1, -1):
                for (work, memory), (cost, mask) in states[k].items():
                    work_next = work + int(q_integer[index])
                    memory_next = memory + int(v_integer[index])
                    if work_next > work_cap or memory_next > memory_cap:
                        continue
                    key = (work_next, memory_next)
                    value = (cost + float(deltas[index]), mask | (1 << index))
                    previous = next_states[k + 1].get(key)
                    if previous is None or value[0] < previous[0] - 1e-12:
                        next_states[k + 1][key] = value
                        states_created += 1
            minimum_k = max(0, target_k - remaining)
            maximum_k = min(target_k, index + 1)
            for k in range(target_k + 1):
                if k < minimum_k or k > maximum_k:
                    next_states[k] = {}
                elif len(next_states[k]) > 1:
                    next_states[k], pruned = _dominance_prune(next_states[k], memory_cap)
                    states_pruned += pruned
            states = next_states
            peak_states = max(peak_states, sum(len(group) for group in states))
        for (work, _memory), (delta_cost, mask) in states[target_k].items():
            workload = work * delta_q
            queue = (
                float(scenario.config["queue_weight"])
                * target_k
                * scenario.queue_delay(workload)
                / float(scenario.config["queue_time_reference_seconds"])
            )
            value = all_local + delta_cost + queue
            if value < best_value - 1e-12:
                best_value = value
                best_strategy = np.array([(mask >> i) & 1 for i in range(scenario.n)], dtype=np.int8)
    runtime = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    if tracing_started:
        tracemalloc.stop()
    reconstructed = scenario.quantized_objective(best_strategy, delta_q, delta_v)
    if abs(reconstructed - best_value) > 1e-7:
        raise AssertionError(f"QDP reconstruction mismatch: {reconstructed} vs {best_value}")
    return QuantizedOracleResult(
        strategy=best_strategy,
        quantized_objective=float(best_value),
        continuous_objective=float(scenario.public_objective(best_strategy)),
        q_integer=q_integer,
        v_integer=v_integer,
        k_offload=int(best_strategy.sum()),
        states_created=states_created,
        states_pruned=states_pruned,
        pruning_rate=states_pruned / states_created if states_created else 0.0,
        peak_live_states=peak_states,
        peak_python_memory_mb=peak_bytes / (1024.0 * 1024.0),
        runtime_seconds=runtime,
    )


def exhaustive_quantized_oracle(
    scenario: Scenario, delta_q: float, delta_v: float
) -> QuantizedOracleResult:
    if scenario.n > 22:
        raise ValueError("Exhaustive audit is limited to N<=22")
    started = time.perf_counter()
    q_integer = np.maximum(np.ceil(scenario.q_estimated / delta_q).astype(int), 1)
    v_integer = np.maximum(np.ceil(scenario.v_estimated / delta_v).astype(int), 1)
    best = math.inf
    best_strategy = np.zeros(scenario.n, dtype=np.int8)
    for mask in range(1 << scenario.n):
        strategy = np.array([(mask >> index) & 1 for index in range(scenario.n)], dtype=np.int8)
        value = scenario.quantized_objective(strategy, delta_q, delta_v)
        if value < best - 1e-12:
            best = value
            best_strategy = strategy
    return QuantizedOracleResult(
        strategy=best_strategy,
        quantized_objective=float(best),
        continuous_objective=float(scenario.public_objective(best_strategy)),
        q_integer=q_integer,
        v_integer=v_integer,
        k_offload=int(best_strategy.sum()),
        states_created=1 << scenario.n,
        states_pruned=0,
        pruning_rate=0.0,
        peak_live_states=1 << scenario.n,
        peak_python_memory_mb=0.0,
        runtime_seconds=time.perf_counter() - started,
    )
