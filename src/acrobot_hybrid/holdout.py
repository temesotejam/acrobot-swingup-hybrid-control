from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .adaptive import AdaptiveLibraryEntry, simulate_adaptive_hybrid
from .evaluation import evaluate_history
from .plant import AcrobotPlant, PhysicsConfig
from .sensing import SensorNoiseConfig


@dataclass(frozen=True)
class HoldoutScenario:
    name: str
    physics: PhysicsConfig


def holdout_physics_models(nominal: PhysicsConfig) -> list[HoldoutScenario]:
    """Parameter values deliberately absent from the 9-model controller bank."""
    return [
        HoldoutScenario(
            "mass/inertia +1.1% holdout",
            replace(
                nominal,
                link_mass_1_kg=nominal.link_mass_1_kg * 1.011,
                link_mass_2_kg=nominal.link_mass_2_kg * 1.011,
                link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 1.011,
                link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 1.011,
            ),
        ),
        HoldoutScenario(
            "mass/inertia -1.3% holdout",
            replace(
                nominal,
                link_mass_1_kg=nominal.link_mass_1_kg * 0.987,
                link_mass_2_kg=nominal.link_mass_2_kg * 0.987,
                link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 0.987,
                link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 0.987,
            ),
        ),
        HoldoutScenario(
            "length/COM +0.6% holdout",
            replace(
                nominal,
                link_length_1_m=nominal.link_length_1_m * 1.006,
                link_length_2_m=nominal.link_length_2_m * 1.006,
                link_com_1_m=nominal.link_com_1_m * 1.006,
                link_com_2_m=nominal.link_com_2_m * 1.006,
            ),
        ),
        HoldoutScenario(
            "length/COM -0.4% holdout",
            replace(
                nominal,
                link_length_1_m=nominal.link_length_1_m * 0.996,
                link_length_2_m=nominal.link_length_2_m * 0.996,
                link_com_1_m=nominal.link_com_1_m * 0.996,
                link_com_2_m=nominal.link_com_2_m * 0.996,
            ),
        ),
        HoldoutScenario(
            "motor tau +6% holdout",
            replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 1.06),
        ),
        HoldoutScenario(
            "motor tau -4% holdout",
            replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 0.96),
        ),
        HoldoutScenario(
            "joint damping +6% holdout",
            replace(
                nominal,
                joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 1.06,
                joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 1.06,
            ),
        ),
        HoldoutScenario(
            "joint damping -7% holdout",
            replace(
                nominal,
                joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 0.93,
                joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 0.93,
            ),
        ),
    ]


def _summarize(rows: list[dict]) -> dict[str, float | int]:
    count = len(rows)
    return {
        "count": count,
        "capture_count": sum(bool(row["capture"]) for row in rows),
        "final_stable_count": sum(bool(row["final_stable"]) for row in rows),
        "capture_rate": float(np.mean([bool(row["capture"]) for row in rows])) if rows else 0.0,
        "final_stable_rate": float(np.mean([bool(row["final_stable"]) for row in rows])) if rows else 0.0,
        "mean_stable_ratio": float(np.mean([float(row["stable_ratio"]) for row in rows])) if rows else 0.0,
    }


def evaluate_holdout_robustness(
    nominal: PhysicsConfig,
    library: dict[str, AdaptiveLibraryEntry],
    identification_s: float = 0.5,
    noisy_seeds: tuple[int, ...] = (11, 22, 33),
    sensor_noise: SensorNoiseConfig | None = None,
) -> tuple[list[dict], dict[str, dict[str, float | int]]]:
    """Evaluate interpolation and noisy-state generalization without adding bank entries."""
    sensor_noise = sensor_noise or SensorNoiseConfig()
    rows: list[dict] = []

    for scenario in holdout_physics_models(nominal):
        actual_plant = AcrobotPlant(scenario.physics, wrap_angles=False)
        clean = simulate_adaptive_hybrid(
            actual_plant,
            library,
            identification_s=identification_s,
            total_s=40.0,
        )
        clean_metrics = evaluate_history(
            actual_plant,
            clean.history.times_s,
            clean.history.states,
            clean.history.commands_nm,
        )
        rows.append(
            {
                "scenario": scenario.name,
                "condition": "clean-holdout",
                "sensor_seed": -1,
                "selected_model": clean.selected_model,
                "capture": bool(clean_metrics.capture),
                "capture_time_s": float(clean_metrics.capture_time_s),
                "final_stable": bool(clean_metrics.final_stable),
                "stable_ratio": float(clean_metrics.stable_ratio),
                "final_2s_stable_ratio": float(clean_metrics.final_2s_stable_ratio),
            }
        )

        for seed in noisy_seeds:
            noisy = simulate_adaptive_hybrid(
                actual_plant,
                library,
                identification_s=identification_s,
                total_s=40.0,
                sensor_noise=sensor_noise,
                sensor_seed=seed,
            )
            noisy_metrics = evaluate_history(
                actual_plant,
                noisy.history.times_s,
                noisy.history.states,
                noisy.history.commands_nm,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "condition": "noisy-holdout",
                    "sensor_seed": int(seed),
                    "selected_model": noisy.selected_model,
                    "capture": bool(noisy_metrics.capture),
                    "capture_time_s": float(noisy_metrics.capture_time_s),
                    "final_stable": bool(noisy_metrics.final_stable),
                    "stable_ratio": float(noisy_metrics.stable_ratio),
                    "final_2s_stable_ratio": float(noisy_metrics.final_2s_stable_ratio),
                }
            )

    clean_rows = [row for row in rows if row["condition"] == "clean-holdout"]
    noisy_rows = [row for row in rows if row["condition"] == "noisy-holdout"]
    return rows, {
        "clean-holdout": _summarize(clean_rows),
        "noisy-holdout": _summarize(noisy_rows),
    }
