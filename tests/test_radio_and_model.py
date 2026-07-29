from __future__ import annotations

import numpy as np
import pytest

from mark10.model import Scenario, generate_wireless_instance


def test_task_type_payload_direction(synthetic_tasks, compact_config) -> None:
    wireless = generate_wireless_instance(synthetic_tasks, seed=11, config=compact_config)
    scenario = Scenario(wireless, wireless.q_llm.to_numpy(), compact_config)
    txt = wireless.index[wireless.predict_type.eq("TXT_2_IMG")][0]
    img = wireless.index[wireless.predict_type.eq("IMG_2_IMG")][0]
    paint = wireless.index[wireless.predict_type.eq("INPAINTING")][0]
    assert scenario.input_image_uplink_bits[txt] == 0
    assert scenario.input_image_uplink_bits[img] > 0
    assert scenario.input_image_uplink_bits[paint] > scenario.input_image_uplink_bits[img]
    assert np.all(scenario.downlink_bits > 0)


def test_wireless_seed_changes_channel_not_task_identity(synthetic_tasks, compact_config) -> None:
    first = generate_wireless_instance(synthetic_tasks, seed=1, config=compact_config)
    repeated = generate_wireless_instance(synthetic_tasks, seed=1, config=compact_config)
    second = generate_wireless_instance(synthetic_tasks, seed=2, config=compact_config)
    assert np.allclose(first.distance_m, repeated.distance_m)
    assert not np.allclose(first.distance_m, second.distance_m)
    assert first.task_uid.tolist() == second.task_uid.tolist()


def test_hard_constraints_and_public_objective(synthetic_tasks, compact_config) -> None:
    wireless = generate_wireless_instance(synthetic_tasks, seed=1, config=compact_config)
    hard = Scenario(wireless, wireless.q_llm.to_numpy(), compact_config, memory_mode="hard")
    soft = Scenario(wireless, wireless.q_llm.to_numpy(), compact_config, memory_mode="soft")
    all_offload = np.ones(len(wireless), dtype=np.int8)
    assert not hard.feasible(all_offload)
    strategy = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    assert hard.public_objective(strategy) == pytest.approx(soft.public_objective(strategy))
    assert soft.decision_objective(strategy) >= soft.public_objective(strategy)


def test_metrics_separate_simulation_truth_from_estimates(synthetic_tasks, compact_config) -> None:
    wireless = generate_wireless_instance(synthetic_tasks, seed=3, config=compact_config)
    q_true = wireless.q_llm.to_numpy()
    v_true = wireless.vram_requirement_gb_simulated.to_numpy()
    scenario = Scenario(
        wireless,
        q_true * 0.5,
        compact_config,
        v_estimated=v_true * 0.5,
        q_true=q_true,
        v_true=v_true,
    )
    metrics = scenario.metrics(np.ones(len(wireless), dtype=np.int8))
    assert metrics["estimated_workload"] < metrics["true_workload"]
    assert metrics["estimated_memory_gb"] < metrics["true_memory_gb"]

