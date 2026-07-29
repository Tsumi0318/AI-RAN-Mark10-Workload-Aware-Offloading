from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mark10.io_utils import load_config


@pytest.fixture
def synthetic_tasks() -> pd.DataFrame:
    n = 8
    return pd.DataFrame(
        {
            "task_uid": [f"synthetic-{i}" for i in range(n)],
            "source_row": np.arange(n),
            "task_pool_id": np.ones(n, dtype=int),
            "task_index": np.arange(n),
            "predict_type": ["TXT_2_IMG", "IMG_2_IMG", "INPAINTING", "TXT_2_IMG"] * 2,
            "prompt_length": [30, 120, 90, 50, 60, 160, 110, 40],
            "negative_prompt_length": [0, 10, 5, 0, 3, 15, 5, 0],
            "num_inference_steps": [20, 40, 35, 25, 30, 50, 45, 22],
            "num_images_per_prompt": [1, 1, 1, 2, 1, 1, 1, 1],
            "num_lora": [0, 1, 1, 0, 2, 1, 0, 0],
            "exec_time_seconds": [12, 28, 25, 16, 20, 35, 30, 14],
            "q_llm": [0.6, 1.3, 1.1, 0.8, 1.0, 1.5, 1.2, 0.5],
            "vram_requirement_gb_simulated": [0.2, 0.35, 0.32, 0.22, 0.3, 0.4, 0.36, 0.18],
        }
    )


@pytest.fixture
def compact_config() -> dict:
    config = load_config()
    config.update(
        {
            "cell_radius_min_m": 20.0,
            "cell_radius_max_m": 80.0,
            "shadowing_sigma_db": 2.0,
            "base_workload_capacity": 8.0,
            "memory_available_main_gb": 1.5,
            "max_updates": 1000,
        }
    )
    return config

