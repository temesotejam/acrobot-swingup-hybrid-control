from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .controllers import lqr_command, trajectory_feedback_command, tvlqr_gains, upright_lqr_gain
from .optimization import OptimizationResult, refine_trajectory_for_model
from .plant import AcrobotPlant, PhysicsConfig
from .sensing import NoisyStateSensor, SensorNoiseConfig
from .simulation import SimulationHistory


@dataclass
class AdaptiveLibraryEntry:
    name: str
    plant: AcrobotPlant
    optimization: OptimizationResult
    tvlqr: np.ndarray
    upright_lqr: np.ndarray


@dataclass
class AdaptiveSimulationResult:
    history: SimulationHistory
    selected_model: str
    identification_errors: dict[str, float]
    identification_steps: int


def candidate_physics_models(nominal: PhysicsConfig) -> dict[str, PhysicsConfig]:
    """Uncertainty bank used by the online multiple-model estimator.

    These are controller hypotheses, not privileged access to the simulated
    plant. Selection is made only from observed state transitions.
    """
    return {
        "nominal": nominal,
        "mass/inertia +2%": replace(
            nominal,
            link_mass_1_kg=nominal.link_mass_1_kg * 1.02,
            link_mass_2_kg=nominal.link_mass_2_kg * 1.02,
            link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 1.02,
            link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 1.02,
        ),
        "mass/inertia -2%": replace(
            nominal,
            link_mass_1_kg=nominal.link_mass_1_kg * 0.98,
            link_mass_2_kg=nominal.link_mass_2_kg * 0.98,
            link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * 0.98,
            link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * 0.98,
        ),
        "length/COM +1%": replace(
            nominal,
            link_length_1_m=nominal.link_length_1_m * 1.01,
            link_length_2_m=nominal.link_length_2_m * 1.01,
            link_com_1_m=nominal.link_com_1_m * 1.01,
            link_com_2_m=nominal.link_com_2_m * 1.01,
        ),
        "length/COM -1%": replace(
            nominal,
            link_length_1_m=nominal.link_length_1_m * 0.99,
            link_length_2_m=nominal.link_length_2_m * 0.99,
            link_com_1_m=nominal.link_com_1_m * 0.99,
            link_com_2_m=nominal.link_com_2_m * 0.99,
        ),
        "motor tau +10%": replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 1.10),
        "motor tau -10%": replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * 0.90),
        "joint damping +10%": replace(
            nominal,
            joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 1.10,
            joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 1.10,
        ),
        "joint damping -10%": replace(
            nominal,
            joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * 0.90,
            joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * 0.90,
        ),
    }


class ModelBankEstimator:
    """Select the model that best predicts observed one-step transitions."""

    def __init__(self, models: dict[str, PhysicsConfig]):
        self.plants = {name: AcrobotPlant(config, wrap_angles=False) for name, config in models.items()}
        self.errors = {name: 0.0 for name in models}
        # Accelerations and actuator dynamics carry most parameter information.
        self.weights = np.array([0.25, 0.25, 4.0, 2.0, 1.0], dtype=np.float64)
        self.samples = 0

    def update(self, state: np.ndarray, command_nm: float, next_state: np.ndarray) -> None:
        before = np.asarray(state, dtype=np.float64)
        after = np.asarray(next_state, dtype=np.float64)
        for name, plant in self.plants.items():
            residual = after - plant.step(before, command_nm)
            self.errors[name] += float(np.sum(self.weights * np.square(residual)))
        self.samples += 1

    def selected_model(self) -> str:
        return min(self.errors, key=self.errors.get)

    def normalized_errors(self) -> dict[str, float]:
        denominator = max(1, self.samples)
        return {name: float(value / denominator) for name, value in self.errors.items()}


def build_adaptive_library(
    nominal_plant: AcrobotPlant,
    nominal_optimization: OptimizationResult,
    candidate_torque_limit_nm: float = 0.99,
) -> dict[str, AdaptiveLibraryEntry]:
    """Build constrained nonlinear replans for the uncertainty bank."""
    configs = candidate_physics_models(nominal_plant.config)
    library: dict[str, AdaptiveLibraryEntry] = {}
    for name, config in configs.items():
        plant = AcrobotPlant(config, wrap_angles=False)
        if name == "nominal":
            optimization = nominal_optimization
        else:
            optimization = refine_trajectory_for_model(
                plant,
                nominal_optimization,
                nominal_torque_limit_nm=candidate_torque_limit_nm,
            )
        library[name] = AdaptiveLibraryEntry(
            name=name,
            plant=plant,
            optimization=optimization,
            tvlqr=tvlqr_gains(plant, optimization.states, optimization.commands_nm),
            upright_lqr=upright_lqr_gain(plant),
        )
    return library


def simulate_adaptive_hybrid(
    simulation_plant: AcrobotPlant,
    library: dict[str, AdaptiveLibraryEntry],
    initial_state: np.ndarray | None = None,
    identification_s: float = 0.5,
    total_s: float = 40.0,
    sensor_noise: SensorNoiseConfig | None = None,
    sensor_seed: int = 0,
) -> AdaptiveSimulationResult:
    """Identify the plant online, then switch to its nonlinear replan + TVLQR.

    When `sensor_noise` is supplied, the model estimator and every feedback
    controller use the same noisy/bias-contaminated measured state. True state
    is retained only inside the simulator and for post-run evaluation.
    """
    if "nominal" not in library:
        raise ValueError("adaptive library must contain a nominal entry")
    dt = simulation_plant.config.dt_s
    for entry in library.values():
        if abs(entry.plant.config.dt_s - dt) > 1e-12:
            raise ValueError("all library models must use the simulation dt")

    nominal = library["nominal"]
    models = {name: entry.plant.config for name, entry in library.items()}
    estimator = ModelBankEstimator(models)
    identify_steps = min(
        int(round(identification_s / dt)),
        nominal.optimization.commands_nm.size - 1,
    )
    state = np.zeros(5, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    sensor = None if sensor_noise is None else NoisyStateSensor(sensor_noise, sensor_seed)

    def observe(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64).copy() if sensor is None else sensor.observe(value)

    measured_state = observe(state)
    times: list[float] = []
    states: list[np.ndarray] = []
    commands: list[float] = []
    modes: list[str] = []

    for index in range(identify_steps):
        command = trajectory_feedback_command(
            measured_state,
            nominal.optimization.states[:, index],
            float(nominal.optimization.commands_nm[index]),
            nominal.tvlqr[index],
            simulation_plant.config.max_torque_nm,
        )
        measured_before = measured_state.copy()
        state = simulation_plant.step(state, command)
        measured_state = observe(state)
        estimator.update(measured_before, command, measured_state)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("identify")

    selected_name = estimator.selected_model()
    selected = library[selected_name]
    if selected.optimization.commands_nm.size != nominal.optimization.commands_nm.size:
        raise ValueError("adaptive library trajectories must share one horizon")

    for index in range(identify_steps, selected.optimization.commands_nm.size):
        command = trajectory_feedback_command(
            measured_state,
            selected.optimization.states[:, index],
            float(selected.optimization.commands_nm[index]),
            selected.tvlqr[index],
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measured_state = observe(state)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("adaptive-tvlqr")

    switch_time_s = selected.optimization.commands_nm.size * dt
    total_steps = int(round(total_s / dt))
    while len(states) < total_steps:
        command = lqr_command(
            measured_state,
            selected.optimization.target_state,
            selected.upright_lqr,
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measured_state = observe(state)
        times.append((len(states) + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("adaptive-lqr")

    history = SimulationHistory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        commands_nm=np.asarray(commands, dtype=np.float64),
        modes=np.asarray(modes, dtype="U24"),
        switch_time_s=float(switch_time_s),
    )
    return AdaptiveSimulationResult(
        history=history,
        selected_model=selected_name,
        identification_errors=estimator.normalized_errors(),
        identification_steps=identify_steps,
    )
