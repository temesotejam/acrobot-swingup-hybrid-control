from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .controllers import tvlqr_gains
from .evaluation import evaluate_history
from .optimization import OptimizationResult
from .plant import AcrobotPlant, PhysicsConfig
from .simulation import simulate_optimized_hybrid, simulate_tvlqr_hybrid


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    group: str
    initial_state: np.ndarray
    physics: PhysicsConfig


def default_robustness_scenarios(nominal: PhysicsConfig) -> list[RobustnessScenario]:
    zero = np.zeros(5, dtype=np.float64)
    scenarios: list[RobustnessScenario] = []

    def initial(name: str, state: list[float], group: str = "initial-state") -> None:
        scenarios.append(RobustnessScenario(name, group, np.asarray(state, dtype=np.float64), nominal))

    initial("theta1 +2 deg", [math.radians(2.0), 0.0, 0.0, 0.0, 0.0])
    initial("theta1 -2 deg", [-math.radians(2.0), 0.0, 0.0, 0.0, 0.0])
    initial("theta2 +2 deg", [0.0, math.radians(2.0), 0.0, 0.0, 0.0])
    initial("theta2 -2 deg", [0.0, -math.radians(2.0), 0.0, 0.0, 0.0])
    initial("dtheta1 +0.05 rad/s", [0.0, 0.0, 0.05, 0.0, 0.0])
    initial("dtheta1 -0.05 rad/s", [0.0, 0.0, -0.05, 0.0, 0.0])
    initial("dtheta2 +0.08 rad/s", [0.0, 0.0, 0.0, 0.08, 0.0])
    initial("dtheta2 -0.08 rad/s", [0.0, 0.0, 0.0, -0.08, 0.0])

    def model(name: str, physics: PhysicsConfig) -> None:
        scenarios.append(RobustnessScenario(name, "model-mismatch", zero.copy(), physics))

    model(
        "mass/inertia +2%",
        replace(
            nominal,
            link_mass_1_kg=nominal.link_mass_1_kg * 1.02,
            link_mass_2_kg=nominal.link_mass_2_kg * 1.02,
            link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 1.02,
            link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 1.02,
        ),
    )
    model(
        "mass/inertia -2%",
        replace(
            nominal,
            link_mass_1_kg=nominal.link_mass_1_kg * 0.98,
            link_mass_2_kg=nominal.link_mass_2_kg * 0.98,
            link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 0.98,
            link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 0.98,
        ),
    )
    model(
        "length/COM +1%",
        replace(
            nominal,
            link_length_1_m=nominal.link_length_1_m * 1.01,
            link_length_2_m=nominal.link_length_2_m * 1.01,
            link_com_1_m=nominal.link_com_1_m * 1.01,
            link_com_2_m=nominal.link_com_2_m * 1.01,
        ),
    )
    model(
        "length/COM -1%",
        replace(
            nominal,
            link_length_1_m=nominal.link_length_1_m * 0.99,
            link_length_2_m=nominal.link_length_2_m * 0.99,
            link_com_1_m=nominal.link_com_1_m * 0.99,
            link_com_2_m=nominal.link_com_2_m * 0.99,
        ),
    )
    model("motor tau +10%", replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 1.10))
    model("motor tau -10%", replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 0.90))
    model(
        "joint damping +10%",
        replace(
            nominal,
            joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 1.10,
            joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 1.10,
        ),
    )
    model(
        "joint damping -10%",
        replace(
            nominal,
            joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 0.90,
            joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 0.90,
        ),
    )

    initial("theta1 +5 deg stress", [math.radians(5.0), 0.0, 0.0, 0.0, 0.0], "stress")
    initial("theta1 -5 deg stress", [-math.radians(5.0), 0.0, 0.0, 0.0, 0.0], "stress")
    initial("theta2 +5 deg stress", [0.0, math.radians(5.0), 0.0, 0.0, 0.0], "stress")
    initial("theta2 -5 deg stress", [0.0, -math.radians(5.0), 0.0, 0.0, 0.0], "stress")
    return scenarios


def evaluate_robustness(
    nominal_plant: AcrobotPlant,
    optimization: OptimizationResult,
    hold_s: float = 19.0,
    gains: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    if gains is None:
        gains = tvlqr_gains(nominal_plant, optimization.states, optimization.commands_nm)
    rows: list[dict] = []
    for scenario in default_robustness_scenarios(nominal_plant.config):
        actual_plant = AcrobotPlant(scenario.physics, wrap_angles=False)
        for controller in ("open-loop", "tvlqr"):
            if controller == "open-loop":
                history = simulate_optimized_hybrid(
                    actual_plant,
                    optimization,
                    hold_s=hold_s,
                    initial_state=scenario.initial_state,
                    controller_plant=nominal_plant,
                )
            else:
                history = simulate_tvlqr_hybrid(
                    nominal_plant,
                    actual_plant,
                    optimization,
                    gains=gains,
                    hold_s=hold_s,
                    initial_state=scenario.initial_state,
                )
            metrics = evaluate_history(actual_plant, history.times_s, history.states, history.commands_nm)
            rows.append(
                {
                    "scenario": scenario.name,
                    "group": scenario.group,
                    "controller": controller,
                    "capture": bool(metrics.capture),
                    "capture_time_s": float(metrics.capture_time_s),
                    "final_stable": bool(metrics.final_stable),
                    "stable_ratio": float(metrics.stable_ratio),
                    "final_2s_stable_ratio": float(metrics.final_2s_stable_ratio),
                    "final_tip_height_m": float(metrics.final_tip_height_m),
                    "rms_torque_nm": float(metrics.rms_torque_nm),
                }
            )

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for group in sorted({row["group"] for row in rows}):
        summary[group] = {}
        for controller in ("open-loop", "tvlqr"):
            selected = [row for row in rows if row["group"] == group and row["controller"] == controller]
            summary[group][controller] = {
                "count": len(selected),
                "capture_count": sum(bool(row["capture"]) for row in selected),
                "final_stable_count": sum(bool(row["final_stable"]) for row in selected),
                "capture_rate": float(np.mean([bool(row["capture"]) for row in selected])) if selected else 0.0,
                "final_stable_rate": float(np.mean([bool(row["final_stable"]) for row in selected])) if selected else 0.0,
                "mean_stable_ratio": float(np.mean([float(row["stable_ratio"]) for row in selected])) if selected else 0.0,
            }
    return rows, summary
