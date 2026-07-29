from __future__ import annotations

from itertools import product

from mark10.experiments import (
    binding_shard_paths,
    binding_capacity_pairs,
    expected_main_instances,
    profiling_error_conditions,
)
from mark10.io_utils import load_config


def test_main_protocol_has_fifty_independent_instances() -> None:
    instances = expected_main_instances(load_config())
    assert len(instances) == 50
    assert len(set(instances)) == 50


def test_binding_grid_contains_every_capacity_pair() -> None:
    config = load_config()
    actual = set(binding_capacity_pairs(config))
    expected = set(product([0.2, 0.4, 0.6, 0.8, 1.0], [1.5, 2.0, 3.0, 5.0, 13.0]))
    assert actual == expected


def test_profile_error_protocol_contains_all_signs_and_protections() -> None:
    conditions = profiling_error_conditions(load_config())
    assert len(conditions) == 4 * 4 * 3
    assert {condition[3] for condition in conditions} == {
        "no_margin",
        "fixed_margin",
        "conservative_quantization",
    }


def test_binding_shard_paths_are_pool_specific() -> None:
    run_path, oracle_path = binding_shard_paths(3)
    assert run_path.name == "binding_resource_runs_pool03.csv"
    assert oracle_path.name == "binding_oracle_diagnostics_pool03.csv"
