from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def generate_wireless_instance(tasks: pd.DataFrame, seed: int, config: dict[str, Any]) -> pd.DataFrame:
    frame = tasks.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    minimum = float(config["cell_radius_min_m"])
    maximum = float(config["cell_radius_max_m"])
    radius = np.sqrt(rng.uniform(minimum**2, maximum**2, len(frame)))
    angle = rng.uniform(0.0, 2.0 * math.pi, len(frame))
    shadowing = rng.normal(0.0, float(config["shadowing_sigma_db"]), len(frame))
    path_loss = (
        float(config["path_loss_reference_db"])
        + 10.0 * float(config["path_loss_exponent"])
        * np.log10(radius / float(config["path_loss_reference_m"]))
        + shadowing
    )
    frame["wireless_seed"] = int(seed)
    frame["distance_m"] = radius
    frame["position_x_m"] = radius * np.cos(angle)
    frame["position_y_m"] = radius * np.sin(angle)
    frame["shadowing_db"] = shadowing
    frame["path_loss_db"] = path_loss
    frame["channel_gain"] = 10.0 ** (-path_loss / 10.0)
    return frame


class Scenario:
    def __init__(
        self,
        tasks: pd.DataFrame,
        q_estimated: np.ndarray,
        config: dict[str, Any],
        *,
        v_estimated: np.ndarray | None = None,
        q_true: np.ndarray | None = None,
        v_true: np.ndarray | None = None,
        workload_capacity: float | None = None,
        memory_available_gb: float | None = None,
        memory_mode: str = "hard",
        bandwidth_hz: float | None = None,
        input_image_payload_mb: float | None = None,
        output_image_payload_mb: float | None = None,
        uplink_power_w: float | None = None,
        downlink_power_w: float | None = None,
        path_loss_exponent: float | None = None,
    ):
        if memory_mode not in {"hard", "soft", "none"}:
            raise ValueError(f"Unsupported memory mode: {memory_mode}")
        self.tasks = tasks.reset_index(drop=True).copy()
        self.n = len(self.tasks)
        self.config = config
        self.memory_mode = memory_mode
        self.q_estimated = np.asarray(q_estimated, dtype=float)
        default_v = self.tasks.vram_requirement_gb_simulated.to_numpy(float)
        self.v_estimated = np.asarray(default_v if v_estimated is None else v_estimated, dtype=float)
        self.q_true = np.asarray(self.q_estimated if q_true is None else q_true, dtype=float)
        self.v_true = np.asarray(self.v_estimated if v_true is None else v_true, dtype=float)
        for name, values in {
            "q_estimated": self.q_estimated,
            "v_estimated": self.v_estimated,
            "q_true": self.q_true,
            "v_true": self.v_true,
        }.items():
            if values.shape != (self.n,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"{name} must contain {self.n} finite positive values")
        self.workload_capacity = float(
            config["base_workload_capacity"] if workload_capacity is None else workload_capacity
        )
        self.workload_limit = self.workload_capacity * (1.0 - float(config["workload_safety_epsilon"]))
        self.memory_available_gb = float(
            config["memory_available_main_gb"] if memory_available_gb is None else memory_available_gb
        )
        self.bandwidth_hz = float(config["bandwidth_hz"] if bandwidth_hz is None else bandwidth_hz)
        self.uplink_power_w = float(config["uplink_power_w"] if uplink_power_w is None else uplink_power_w)
        self.downlink_power_w = float(config["downlink_power_w"] if downlink_power_w is None else downlink_power_w)
        self.input_image_payload_mb = float(
            config["input_image_payload_mb"] if input_image_payload_mb is None else input_image_payload_mb
        )
        self.output_image_payload_mb = float(
            config["output_image_payload_mb"] if output_image_payload_mb is None else output_image_payload_mb
        )
        self.path_loss_exponent = float(
            config["path_loss_exponent"] if path_loss_exponent is None else path_loss_exponent
        )
        self._refresh_channel_for_path_loss_override()
        self._build_payloads()
        self._precompute_costs()

    def _refresh_channel_for_path_loss_override(self) -> None:
        if self.path_loss_exponent == float(self.config["path_loss_exponent"]):
            self.channel_gain = self.tasks.channel_gain.to_numpy(float)
            return
        path_loss = (
            float(self.config["path_loss_reference_db"])
            + 10.0 * self.path_loss_exponent
            * np.log10(self.tasks.distance_m.to_numpy(float) / float(self.config["path_loss_reference_m"]))
            + self.tasks.shadowing_db.to_numpy(float)
        )
        self.channel_gain = 10.0 ** (-path_loss / 10.0)

    def _build_payloads(self) -> None:
        characters = (
            self.tasks.prompt_length.to_numpy(float)
            + self.tasks.negative_prompt_length.to_numpy(float)
        )
        images = self.tasks.num_images_per_prompt.to_numpy(float)
        prompt_metadata_bytes = (
            float(self.config["metadata_bytes"])
            + float(self.config["bytes_per_character"]) * characters
        )
        types = self.tasks.predict_type.astype(str).to_numpy()
        input_mb = np.where(types == "IMG_2_IMG", self.input_image_payload_mb * images, 0.0)
        input_mb = np.where(
            types == "INPAINTING",
            (self.input_image_payload_mb + float(self.config["mask_payload_mb"])) * images,
            input_mb,
        )
        self.metadata_prompt_bits = prompt_metadata_bytes * 8.0
        self.input_image_uplink_bits = input_mb * 1024.0 * 1024.0 * 8.0
        self.uplink_bits = self.metadata_prompt_bits + self.input_image_uplink_bits
        self.downlink_bits = images * self.output_image_payload_mb * 1024.0 * 1024.0 * 8.0

    def _precompute_costs(self) -> None:
        noise = 10.0 ** ((float(self.config["noise_density_dbm_hz"]) - 30.0) / 10.0)
        noise *= 10.0 ** (float(self.config["noise_figure_db"]) / 10.0)
        self.tx_time_seconds = np.zeros((self.n + 1, self.n), dtype=float)
        self.tx_energy_j = np.zeros((self.n + 1, self.n), dtype=float)
        execution = self.tasks.exec_time_seconds.to_numpy(float)
        median_execution = max(float(np.median(execution)), 1e-12)
        self.edge_compute_cost = execution / median_execution
        self.local_execution_seconds = float(self.config["local_slowdown"]) * execution
        self.local_cost = (
            float(self.config["local_slowdown"]) * self.edge_compute_cost
            + float(self.config["local_energy_addition"])
        )
        self.local_energy_j = float(self.config["local_device_power_w"]) * self.local_execution_seconds
        for k in range(1, self.n + 1):
            bandwidth = self.bandwidth_hz / k
            snr_up = self.uplink_power_w * self.channel_gain / (noise * bandwidth)
            snr_down = self.downlink_power_w * self.channel_gain / (noise * bandwidth)
            rate_up = bandwidth * np.log2(1.0 + snr_up)
            rate_down = bandwidth * np.log2(1.0 + snr_down)
            up_time = self.uplink_bits / rate_up
            down_time = self.downlink_bits / rate_down
            self.tx_time_seconds[k] = up_time + down_time
            self.tx_energy_j[k] = (
                self.uplink_power_w * up_time
                + float(self.config["downlink_receive_power_w"]) * down_time
            )
        self.offload_base = np.zeros((self.n + 1, self.n), dtype=float)
        for k in range(1, self.n + 1):
            self.offload_base[k] = (
                self.edge_compute_cost
                + float(self.config["wireless_time_weight"])
                * self.tx_time_seconds[k]
                / float(self.config["radio_time_reference_seconds"])
                + float(self.config["wireless_energy_weight"])
                * self.tx_energy_j[k]
                / float(self.config["radio_energy_reference_j"])
            )

    def queue_delay(self, workload: float) -> float:
        rho = workload / self.workload_capacity
        if rho >= 1.0:
            return math.inf
        return float(self.config["queue_delay_base_seconds"]) + float(
            self.config["queue_delay_scale_seconds"]
        ) * rho / max(1.0 - rho, 1e-12)

    def state_values(self, strategy: np.ndarray, *, truth: bool = False) -> tuple[int, float, float]:
        selected = np.asarray(strategy, dtype=bool)
        q = self.q_true if truth else self.q_estimated
        v = self.v_true if truth else self.v_estimated
        return int(selected.sum()), float(q[selected].sum()), float(v[selected].sum())

    def feasible(self, strategy: np.ndarray) -> bool:
        _, workload, memory = self.state_values(strategy)
        workload_ok = workload <= self.workload_limit + 1e-12
        memory_ok = memory <= self.memory_available_gb + 1e-12
        return workload_ok and (memory_ok or self.memory_mode != "hard")

    def public_objective(self, strategy: np.ndarray, *, enforce_workload: bool = True) -> float:
        selected = np.asarray(strategy, dtype=bool)
        k = int(selected.sum())
        workload = float(self.q_estimated[selected].sum())
        if enforce_workload and workload > self.workload_limit + 1e-12:
            return math.inf
        total = float(self.local_cost[~selected].sum())
        if k:
            total += float(self.offload_base[k, selected].sum())
            total += (
                float(self.config["queue_weight"])
                * k
                * self.queue_delay(workload)
                / float(self.config["queue_time_reference_seconds"])
            )
        return total

    def memory_barrier(self, memory_gb: float) -> float:
        utilization = memory_gb / self.memory_available_gb
        exponent = float(np.clip(float(self.config["soft_barrier_beta"]) * (utilization - 1.0), -30, 30))
        return float(self.config["soft_barrier_alpha"]) * math.exp(exponent)

    def decision_objective(self, strategy: np.ndarray) -> float:
        if not self.feasible(strategy):
            return math.inf
        value = self.public_objective(strategy)
        if self.memory_mode == "soft":
            selected = np.asarray(strategy, dtype=bool)
            k = int(selected.sum())
            memory = float(self.v_estimated[selected].sum())
            value += k * self.memory_barrier(memory) if k else 0.0
        return value

    def quantized_objective(self, strategy: np.ndarray, delta_q: float, delta_v: float) -> float:
        selected = np.asarray(strategy, dtype=bool)
        k = int(selected.sum())
        q_integer = np.maximum(np.ceil(self.q_estimated / delta_q).astype(int), 1)
        v_integer = np.maximum(np.ceil(self.v_estimated / delta_v).astype(int), 1)
        workload = float(q_integer[selected].sum()) * delta_q
        memory = float(v_integer[selected].sum()) * delta_v
        if workload > self.workload_limit + 1e-12 or memory > self.memory_available_gb + 1e-12:
            return math.inf
        total = float(self.local_cost[~selected].sum())
        if k:
            total += float(self.offload_base[k, selected].sum())
            total += (
                float(self.config["queue_weight"])
                * k
                * self.queue_delay(workload)
                / float(self.config["queue_time_reference_seconds"])
            )
        return total

    def metrics(self, strategy: np.ndarray) -> dict[str, Any]:
        selected = np.asarray(strategy, dtype=bool)
        k, workload, memory = self.state_values(strategy)
        _, true_workload, true_memory = self.state_values(strategy, truth=True)
        queue = self.queue_delay(true_workload) if k else 0.0
        local_delay = float(self.local_execution_seconds[~selected].sum())
        offload_delay = 0.0
        if k:
            offload_delay = float(
                (
                    self.tx_time_seconds[k, selected]
                    + self.tasks.exec_time_seconds.to_numpy(float)[selected]
                    + queue
                ).sum()
            )
        total_energy = float(self.local_energy_j[~selected].sum())
        if k:
            total_energy += float(self.tx_energy_j[k, selected].sum())
        barrier_penalty = 0.0
        if self.memory_mode == "soft" and k:
            barrier_penalty = k * self.memory_barrier(memory)
        return {
            "n_nodes": self.n,
            "k_offload": k,
            "offload_rate": k / self.n,
            "decision_objective": self.decision_objective(strategy),
            "public_objective_J": self.public_objective(strategy),
            "barrier_penalty": barrier_penalty,
            "estimated_workload": workload,
            "true_workload": true_workload,
            "workload_capacity": self.workload_capacity,
            "workload_limit": self.workload_limit,
            "workload_utilization": true_workload / self.workload_capacity,
            "estimated_memory_gb": memory,
            "true_memory_gb": true_memory,
            "memory_available_gb": self.memory_available_gb,
            "memory_utilization": true_memory / self.memory_available_gb,
            "estimated_workload_violation": int(workload > self.workload_limit + 1e-12),
            "estimated_memory_violation": int(memory > self.memory_available_gb + 1e-12),
            "true_workload_violation": int(true_workload > self.workload_limit + 1e-12),
            "true_memory_violation": int(true_memory > self.memory_available_gb + 1e-12),
            "total_end_to_end_delay_seconds": local_delay + offload_delay,
            "mean_end_to_end_delay_seconds": (local_delay + offload_delay) / self.n,
            "total_device_energy_j": total_energy,
            "mean_tx_time_ms_offloaded": float(self.tx_time_seconds[k, selected].mean() * 1000.0) if k else 0.0,
            "mean_tx_energy_mj_offloaded": float(self.tx_energy_j[k, selected].mean() * 1000.0) if k else 0.0,
            "queue_delay_ms_per_offloaded_task": queue * 1000.0,
        }

