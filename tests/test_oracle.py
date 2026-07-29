from __future__ import annotations

import numpy as np
import pytest

from mark10.algorithms import exhaustive_quantized_oracle, quantized_oracle
from mark10.model import Scenario, generate_wireless_instance


def test_qdp_uses_upward_quantization(synthetic_tasks, compact_config) -> None:
    wireless = generate_wireless_instance(synthetic_tasks, seed=23, config=compact_config)
    scenario = Scenario(wireless, wireless.q_llm.to_numpy(), compact_config)
    result = quantized_oracle(scenario, delta_q=0.1, delta_v=0.1)
    assert np.array_equal(result.q_integer, np.ceil(scenario.q_estimated / 0.1).astype(int))
    assert np.array_equal(result.v_integer, np.ceil(scenario.v_estimated / 0.1).astype(int))


def test_qdp_matches_exhaustive_quantized_optimum(synthetic_tasks, compact_config) -> None:
    wireless = generate_wireless_instance(synthetic_tasks.iloc[:7], seed=29, config=compact_config)
    scenario = Scenario(wireless, wireless.q_llm.to_numpy(), compact_config)
    qdp = quantized_oracle(scenario, 0.1, 0.1)
    brute = exhaustive_quantized_oracle(scenario, 0.1, 0.1)
    assert qdp.quantized_objective == pytest.approx(brute.quantized_objective)
    assert scenario.feasible(qdp.strategy)
