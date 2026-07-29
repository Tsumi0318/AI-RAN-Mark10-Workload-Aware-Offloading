from __future__ import annotations

import numpy as np

from mark10.algorithms import (
    benefit_resource_repair,
    full_single_flip_check,
    lagrangian_relaxation,
    run_wa_mcbr,
    signaling_bytes_per_update,
    swap_improve,
)
from mark10.model import Scenario, generate_wireless_instance


def _scenario(tasks, config) -> Scenario:
    wireless = generate_wireless_instance(tasks, seed=17, config=config)
    return Scenario(wireless, wireless.q_llm.to_numpy(), config)


def test_repair_returns_jointly_feasible_strategy(synthetic_tasks, compact_config) -> None:
    scenario = _scenario(synthetic_tasks, compact_config)
    repaired = benefit_resource_repair(scenario, np.ones(scenario.n, dtype=np.int8))
    assert scenario.feasible(repaired)


def test_accepted_updates_strictly_reduce_objective(synthetic_tasks, compact_config) -> None:
    scenario = _scenario(synthetic_tasks, compact_config)
    result = run_wa_mcbr(
        scenario,
        seed=7,
        record_trace=True,
        initial_strategy=np.zeros(scenario.n, dtype=np.int8),
    )
    accepted = [row for row in result.trace if row["accepted"]]
    assert accepted
    assert all(
        row["objective_after"] < row["objective_before"] - compact_config["epsilon_algorithm"]
        for row in accepted
    )
    assert result.full_single_flip_pass
    assert full_single_flip_check(scenario, result.strategy, compact_config["epsilon_algorithm"])


def test_swap_and_lagrangian_return_feasible_strategies(synthetic_tasks, compact_config) -> None:
    scenario = _scenario(synthetic_tasks, compact_config)
    base = run_wa_mcbr(scenario, seed=9)
    swapped, _ = swap_improve(scenario, base.strategy)
    lagrangian = lagrangian_relaxation(scenario)
    assert scenario.feasible(swapped)
    assert scenario.feasible(lagrangian.strategy)


def test_signaling_bytes_include_payload_and_headers(compact_config) -> None:
    details = signaling_bytes_per_update(compact_config)
    assert details["payload_bytes"] == 38
    assert details["header_bytes"] == 80
    assert details["total_bytes"] == 118
