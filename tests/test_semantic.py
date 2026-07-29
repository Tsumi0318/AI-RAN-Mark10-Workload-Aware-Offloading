from __future__ import annotations

import json

import pandas as pd
import pytest

from mark10.semantic import SemanticProfile, build_intent, parse_profile, run_fresh_profiles


@pytest.fixture
def task_row() -> pd.Series:
    return pd.Series(
        {
            "task_uid": "pool01-task001",
            "predict_type": "IMG_2_IMG",
            "prompt_length": 150,
            "negative_prompt_length": 20,
            "num_inference_steps": 50,
            "num_images_per_prompt": 1,
            "num_lora": 2,
            "exec_time_seconds": 41.2,
        }
    )


def test_intent_does_not_leak_execution_time(task_row: pd.Series) -> None:
    intent = build_intent(task_row)
    serialized = json.dumps(intent).lower()
    assert "exec_time" not in serialized
    assert "execution_time" not in serialized
    assert "41.2" not in serialized


def test_intent_contains_required_semantic_features(task_row: pd.Series) -> None:
    intent = build_intent(task_row)
    assert intent == {
        "task_uid": "pool01-task001",
        "task_type": "IMG_2_IMG",
        "prompt_length_chars": 150,
        "negative_prompt_length_chars": 20,
        "steps": 50,
        "num_images": 1,
        "lora_count": 2,
    }


def test_parser_accepts_strict_valid_json() -> None:
    parsed = parse_profile(
        '{"compute_multiplier":1.2,"memory_multiplier":0.9,'
        '"semantic_class":"moderate","warning":"normal"}'
    )
    assert parsed == SemanticProfile(1.2, 0.9, "moderate", "normal")


@pytest.mark.parametrize("field", ["compute_multiplier", "memory_multiplier"])
def test_parser_rejects_out_of_range_multiplier(field: str) -> None:
    payload = {
        "compute_multiplier": 1.0,
        "memory_multiplier": 1.0,
        "semantic_class": "moderate",
        "warning": "normal",
    }
    payload[field] = 9.0
    with pytest.raises(ValueError, match=field):
        parse_profile(json.dumps(payload))


def test_parser_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_profile('{"compute_multiplier":1.0}')


def test_parser_normalizes_empty_warning_to_explicit_none() -> None:
    parsed = parse_profile(
        '{"compute_multiplier":1.0,"memory_multiplier":1.0,'
        '"semantic_class":"light","warning":""}'
    )
    assert parsed.warning == "none_reported"


def test_fresh_profiles_checkpoint_successes_before_late_failure(tmp_path) -> None:
    class LateFailureClient:
        def evaluate(self, intent):
            if intent["task_uid"].endswith("002"):
                raise RuntimeError("late failure")
            return {
                "compute_multiplier": 1.0,
                "memory_multiplier": 1.0,
                "semantic_class": "baseline",
                "warning": "none_reported",
                "requested_model": "test",
                "resolved_model": "test",
                "latency_ms": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "attempts": 1,
                "response_sha256": "0" * 64,
            }

    tasks = pd.DataFrame(
        [
            {
                "task_uid": f"pool01-task{index:03d}",
                "predict_type": "TXT_2_IMG",
                "prompt_length": 10,
                "negative_prompt_length": 0,
                "num_inference_steps": 20,
                "num_images_per_prompt": 1,
                "num_lora": 0,
            }
            for index in range(3)
        ]
    )
    config = {
        "deepseek_workers": 1,
        "n_task_pools": 1,
        "tasks_per_pool": 3,
        "deepseek_temperature": 0,
        "deepseek_model": "test",
    }
    checkpoint = tmp_path / "checkpoint.csv"
    with pytest.raises(RuntimeError, match="late failure"):
        run_fresh_profiles(tasks, LateFailureClient(), config, checkpoint_path=checkpoint)
    saved = pd.read_csv(checkpoint)
    assert saved.task_uid.tolist() == ["pool01-task000", "pool01-task001"]
