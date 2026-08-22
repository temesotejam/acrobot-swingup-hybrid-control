from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .adaptive import AdaptiveLibraryEntry
from .evaluation import evaluate_history
from .plant import AcrobotPlant, PhysicsConfig
from .sensing import SensorNoiseConfig
from .terminal_replan import simulate_terminal_replan_hybrid


@dataclass(frozen=True)
class HoldoutScenario:
    name: str
    physics: PhysicsConfig


def holdout_physics_models(nominal: PhysicsConfig) -> list[HoldoutScenario]:
    """Parameter values deliberately absent from the 9-model controller bank."""
    return [
        HoldoutScenario("mass/inertia +1.1% holdout", replace(nominal, link_mass_1_kg=nominal.link_mass_1_kg * 1.011, link_mass_2_kg=nominal.link_mass_2_kg * 1.011, link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 1.011, link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 1.011)),
        HoldoutScenario("mass/inertia -1.3% holdout", replace(nominal, link_mass_1_kg=nominal.link_mass_1_kg * 0.987, link_mass_2_kg=nominal.link_mass_2_kg * 0.987, link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 0.987, link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 0.987)),
        HoldoutScenario("length/COM +0.6% holdout", replace(nominal, link_length_1_m=nominal.link_length_1_m * 1.006, link_length_2_m=nominal.link_length_2_m * 1.006, link_com_1_m=nominal.link_com_1_m * 1.006, link_com_2_m=nominal.link_com_2_m * 1.006)),
        HoldoutScenario("length/COM -0.4% holdout", replace(nominal, link_length_1_m=nominal.link_length_1_m * 0.996, link_length_2_m=nominal.link_length_2_m * 0.996, link_com_1_m=nominal.link_com_1_m * 0.996, link_com_2_m=nominal.link_com_2_m * 0.996)),
        HoldoutScenario("motor tau +6% holdout", replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 1.06)),
        HoldoutScenario("motor tau -4% holdout", replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 0.96)),
        HoldoutScenario("joint damping +6% holdout", replace(nominal, joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 1.06, joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 1.06)),
        HoldoutScenario("joint damping -7% holdout", replace(nominal, joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 0.93, joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 0.93)),
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


def _result_row(
    scenario: HoldoutScenario,
    condition: str,
    sensor_seed: int,
    plant: AcrobotPlant,
    result,
) -> dict:
    metrics = evaluate_history(plant, result.history.times_s, result.history.states, result.history.commands_nm)
    return {
        "scenario": scenario.name,
        "condition": condition,
        "sensor_seed": int(sensor_seed),
        "selected_model": result.selected_model,
        "capture": bool(metrics.capture),
        "capture_time_s": float(metrics.capture_time_s),
        "final_stable": bool(metrics.final_stable),
        "stable_ratio": float(metrics.stable_ratio),
        "final_2s_stable_ratio": float(metrics.final_2s_stable_ratio),
    }


def evaluate_holdout_robustness(
    nominal: PhysicsConfig,
    library: dict[str, AdaptiveLibraryEntry],
    identification_s: float = 1.0,
    noisy_seeds: tuple[int, ...] = (11, 22, 33),
    sensor_noise: SensorNoiseConfig | None = None,
) -> tuple[list[dict], dict[str, dict[str, float | int]]]:
    """Evaluate bank-interior parameters and diagnose noisy upright hold.

    The primary noisy trials use a conservative state-and-dwell handoff to the
    EKF-backed upright LQR.  One representative holdout family is additionally
    replayed with a fixed end-of-trajectory handoff and three hold-state
    sources.  That separates a bad handoff from estimator error and from a
    regulator that cannot stabilize the actual plant even with perfect state.
    """
    sensor_noise = sensor_noise or SensorNoiseConfig()
    rows: list[dict] = []
    diagnostic_scenario = "motor tau +6% holdout"

    for scenario in holdout_physics_models(nominal):
        actual_plant = AcrobotPlant(scenario.physics, wrap_angles=False)
        clean = simulate_terminal_replan_hybrid(
            actual_plant,
            library,
            identification_s=max(1.0, float(identification_s)),
            replan_start_s=18.0,
            total_s=40.0,
        )
        rows.append(_result_row(scenario, "clean-holdout", -1, actual_plant, clean))

        for seed in noisy_seeds:
            noisy = simulate_terminal_replan_hybrid(
                actual_plant,
                library,
                identification_s=max(1.0, float(identification_s)),
                replan_start_s=18.0,
                total_s=40.0,
                sensor_noise=sensor_noise,
                sensor_seed=seed,
                calibration_samples=50,
                hold_state_source="ekf",
                capture_supervisor=True,
                capture_supervisor_start_s=18.0,
                capture_supervisor_dwell_s=0.20,
            )
            noisy_row = _result_row(scenario, "noisy-holdout", seed, actual_plant, noisy)
            rows.append(noisy_row)

            if scenario.name == diagnostic_scenario:
                supervised_row = dict(noisy_row)
                supervised_row["condition"] = "diagnostic-ekf-supervised"
                rows.append(supervised_row)

                for hold_source in ("true", "measurement", "ekf"):
                    diagnostic = simulate_terminal_replan_hybrid(
                        actual_plant,
                        library,
                        identification_s=max(1.0, float(identification_s)),
                        replan_start_s=18.0,
                        total_s=40.0,
                        sensor_noise=sensor_noise,
                        sensor_seed=seed,
                        calibration_samples=50,
                        hold_state_source=hold_source,
                        capture_supervisor=False,
                    )
                    rows.append(
                        _result_row(
                            scenario,
                            f"diagnostic-{hold_source}-fixed-handoff",
                            seed,
                            actual_plant,
                            diagnostic,
                        )
                    )

    conditions = sorted({str(row["condition"]) for row in rows})
    summary = {
        condition: _summarize([row for row in rows if row["condition"] == condition])
        for condition in conditions
    }
    return rows, summary
