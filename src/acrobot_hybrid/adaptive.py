from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .controllers import (
    lqr_command,
    trajectory_feedback_command,
    tvlqr_gains,
    upright_discrete_lqr_gain,
    upright_lqr_gain,
)
from .optimization import OptimizationResult, refine_trajectory_for_model
from .plant import AcrobotPlant, PhysicsConfig
from .sensing import ExtendedKalmanObserver, ModelPredictiveObserver, NoisyStateSensor, SensorNoiseConfig
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


@dataclass(frozen=True)
class ContinuousModelEstimate:
    family: str
    alpha: float
    error: float
    physics: PhysicsConfig


FAMILY_ENDPOINTS = {
    "mass/inertia": ("mass/inertia -2%", "mass/inertia +2%"),
    "length/COM": ("length/COM -1%", "length/COM +1%"),
    "motor tau": ("motor tau -10%", "motor tau +10%"),
    "joint damping": ("joint damping -10%", "joint damping +10%"),
}


def interpolate_physics(nominal: PhysicsConfig, family: str, alpha: float) -> PhysicsConfig:
    alpha = float(np.clip(alpha, -1.0, 1.0))
    if family == "mass/inertia":
        scale = 1.0 + 0.02 * alpha
        return replace(
            nominal,
            link_mass_1_kg=nominal.link_mass_1_kg * scale,
            link_mass_2_kg=nominal.link_mass_2_kg * scale,
            link_moi_1_kg_m2=nominal.link_moi_1_kg_m2 * scale,
            link_moi_2_kg_m2=nominal.link_moi_2_kg_m2 * scale,
        )
    if family == "length/COM":
        scale = 1.0 + 0.01 * alpha
        return replace(
            nominal,
            link_length_1_m=nominal.link_length_1_m * scale,
            link_length_2_m=nominal.link_length_2_m * scale,
            link_com_1_m=nominal.link_com_1_m * scale,
            link_com_2_m=nominal.link_com_2_m * scale,
        )
    if family == "motor tau":
        return replace(nominal, motor_time_constant_s=nominal.motor_time_constant_s * (1.0 + 0.10 * alpha))
    if family == "joint damping":
        scale = 1.0 + 0.10 * alpha
        return replace(
            nominal,
            joint1_damping_nm_per_rad_s=nominal.joint1_damping_nm_per_rad_s * scale,
            joint2_damping_nm_per_rad_s=nominal.joint2_damping_nm_per_rad_s * scale,
        )
    raise ValueError(f"unknown uncertainty family: {family}")


def candidate_physics_models(nominal: PhysicsConfig) -> dict[str, PhysicsConfig]:
    return {
        "nominal": nominal,
        "mass/inertia +2%": interpolate_physics(nominal, "mass/inertia", +1.0),
        "mass/inertia -2%": interpolate_physics(nominal, "mass/inertia", -1.0),
        "length/COM +1%": interpolate_physics(nominal, "length/COM", +1.0),
        "length/COM -1%": interpolate_physics(nominal, "length/COM", -1.0),
        "motor tau +10%": interpolate_physics(nominal, "motor tau", +1.0),
        "motor tau -10%": interpolate_physics(nominal, "motor tau", -1.0),
        "joint damping +10%": interpolate_physics(nominal, "joint damping", +1.0),
        "joint damping -10%": interpolate_physics(nominal, "joint damping", -1.0),
    }


class ModelBankEstimator:
    """Estimate dynamics from a multi-step observed command/state window."""

    def __init__(self, models: dict[str, PhysicsConfig]):
        self.plants = {name: AcrobotPlant(config, wrap_angles=False) for name, config in models.items()}
        self.weights = np.array([0.05, 0.05, 4.0, 2.0, 0.5], dtype=np.float64)
        self.initial_state: np.ndarray | None = None
        self.commands: list[float] = []
        self.observations: list[np.ndarray] = []
        self.samples = 0

    def update(self, state: np.ndarray, command_nm: float, next_state: np.ndarray) -> None:
        before = np.asarray(state, dtype=np.float64)
        after = np.asarray(next_state, dtype=np.float64)
        if self.initial_state is None:
            self.initial_state = before.copy()
            self.observations = [before.copy()]
        self.commands.append(float(command_nm))
        self.observations.append(after.copy())
        self.samples += 1

    def _rollout_error(self, plant: AcrobotPlant) -> float:
        if self.initial_state is None or not self.commands:
            return 0.0
        predicted = self.initial_state.copy()
        initial_observation = self.observations[0]
        total = 0.0
        for command, observed in zip(self.commands, self.observations[1:], strict=True):
            predicted = plant.step(predicted, command)
            residual = (observed - initial_observation) - (predicted - self.initial_state)
            total += float(np.sum(self.weights * np.square(residual)))
        return total

    def _rollout_errors(self) -> dict[str, float]:
        return {name: self._rollout_error(plant) for name, plant in self.plants.items()}

    def selected_model(self) -> str:
        errors = self._rollout_errors()
        return min(errors, key=errors.get)

    def normalized_errors(self) -> dict[str, float]:
        denominator = max(1, self.samples)
        return {name: float(value / denominator) for name, value in self._rollout_errors().items()}

    def continuous_estimate(self, nominal: PhysicsConfig, grid_points: int = 41) -> ContinuousModelEstimate:
        best: ContinuousModelEstimate | None = None
        for family in FAMILY_ENDPOINTS:
            for alpha in np.linspace(-1.0, 1.0, int(grid_points)):
                physics = interpolate_physics(nominal, family, float(alpha))
                error = self._rollout_error(AcrobotPlant(physics, wrap_angles=False))
                candidate = ContinuousModelEstimate(family, float(alpha), float(error), physics)
                if best is None or candidate.error < best.error:
                    best = candidate
        if best is None:
            raise RuntimeError("continuous estimator has no candidates")
        return best


def build_adaptive_library(
    nominal_plant: AcrobotPlant,
    nominal_optimization: OptimizationResult,
    candidate_torque_limit_nm: float = 0.99,
) -> dict[str, AdaptiveLibraryEntry]:
    library: dict[str, AdaptiveLibraryEntry] = {}
    for name, config in candidate_physics_models(nominal_plant.config).items():
        plant = AcrobotPlant(config, wrap_angles=False)
        optimization = (
            nominal_optimization
            if name == "nominal"
            else refine_trajectory_for_model(
                plant,
                nominal_optimization,
                nominal_torque_limit_nm=candidate_torque_limit_nm,
            )
        )
        library[name] = AdaptiveLibraryEntry(
            name=name,
            plant=plant,
            optimization=optimization,
            tvlqr=tvlqr_gains(plant, optimization.states, optimization.commands_nm),
            upright_lqr=upright_lqr_gain(plant),
        )
    return library


def _interpolated_library_entry(
    library: dict[str, AdaptiveLibraryEntry],
    estimate: ContinuousModelEstimate,
) -> AdaptiveLibraryEntry:
    nominal = library["nominal"]
    negative_name, positive_name = FAMILY_ENDPOINTS[estimate.family]
    endpoint = library[positive_name if estimate.alpha >= 0.0 else negative_name]
    weight = abs(float(estimate.alpha))
    optimization = OptimizationResult(
        states=(1.0 - weight) * nominal.optimization.states + weight * endpoint.optimization.states,
        commands_nm=(1.0 - weight) * nominal.optimization.commands_nm + weight * endpoint.optimization.commands_nm,
        target_state=nominal.optimization.target_state.copy(),
        objective=float("nan"),
        solver_status="interpolated",
        nominal_torque_limit_nm=(1.0 - weight) * nominal.optimization.nominal_torque_limit_nm
        + weight * endpoint.optimization.nominal_torque_limit_nm,
    )
    plant = AcrobotPlant(estimate.physics, wrap_angles=False)
    return AdaptiveLibraryEntry(
        name=f"{estimate.family} alpha={estimate.alpha:+.3f}",
        plant=plant,
        optimization=optimization,
        tvlqr=(1.0 - weight) * nominal.tvlqr + weight * endpoint.tvlqr,
        upright_lqr=(1.0 - weight) * nominal.upright_lqr + weight * endpoint.upright_lqr,
    )


def _capture_ready(plant: AcrobotPlant, estimated_state: np.ndarray) -> bool:
    """Conservative local-LQR entry set used by the state supervisor."""
    return bool(
        plant.tip_height_m(estimated_state) >= 1.45
        and abs(float(estimated_state[2])) <= 0.50
        and abs(float(estimated_state[3])) <= 0.80
    )


def simulate_adaptive_hybrid(
    simulation_plant: AcrobotPlant,
    library: dict[str, AdaptiveLibraryEntry],
    initial_state: np.ndarray | None = None,
    identification_s: float = 0.5,
    total_s: float = 40.0,
    sensor_noise: SensorNoiseConfig | None = None,
    sensor_seed: int = 0,
    continuous_selection: bool = False,
    calibration_samples: int = 50,
    capture_supervisor: bool = False,
    capture_supervisor_start_s: float = 15.0,
    capture_supervisor_dwell_s: float = 0.20,
) -> AdaptiveSimulationResult:
    """Identify dynamics, follow TVLQR, then hand off by time or capture state.

    Endpoint regression tests leave `capture_supervisor=False` and therefore
    preserve the historical fixed 21 s handoff. Holdout/noisy tests enable the
    supervisor, which enters the upright regulator as soon as a filtered state
    remains inside a conservative capture set for a short dwell.
    """
    if "nominal" not in library:
        raise ValueError("adaptive library must contain a nominal entry")
    dt = simulation_plant.config.dt_s
    nominal = library["nominal"]
    estimator = ModelBankEstimator({name: entry.plant.config for name, entry in library.items()})
    identify_steps = min(int(round(identification_s / dt)), nominal.optimization.commands_nm.size - 1)
    state = np.zeros(5, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    sensor = None if sensor_noise is None else NoisyStateSensor(sensor_noise, sensor_seed)

    bias_estimate = np.zeros(5, dtype=np.float64)
    if sensor is not None and calibration_samples > 0:
        samples = np.asarray([sensor.observe(state) for _ in range(int(calibration_samples))])
        bias_estimate = np.mean(samples, axis=0) - state

    def observe(value: np.ndarray) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64).copy() if sensor is None else sensor.observe(value)
        return raw - bias_estimate

    measurement = observe(state)
    swing_observer = None if sensor is None else ModelPredictiveObserver(nominal.plant, measurement)
    control_state = measurement.copy() if swing_observer is None else swing_observer.estimate.copy()
    times: list[float] = []
    states: list[np.ndarray] = []
    commands: list[float] = []
    modes: list[str] = []

    for index in range(identify_steps):
        command = trajectory_feedback_command(
            control_state,
            nominal.optimization.states[:, index],
            float(nominal.optimization.commands_nm[index]),
            nominal.tvlqr[index],
            simulation_plant.config.max_torque_nm,
        )
        measured_before = measurement.copy()
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        estimator.update(measured_before, command, measurement)
        control_state = measurement.copy() if swing_observer is None else swing_observer.update(command, measurement)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("identify")

    if continuous_selection:
        estimate = estimator.continuous_estimate(nominal.plant.config)
        selected = _interpolated_library_entry(library, estimate)
        selected_name = selected.name
    else:
        selected_name = estimator.selected_model()
        selected = library[selected_name]
    if swing_observer is not None:
        swing_observer.set_plant(selected.plant)

    required_capture_steps = max(1, int(round(capture_supervisor_dwell_s / dt)))
    capture_streak = 0
    handoff_step: int | None = None
    for index in range(identify_steps, selected.optimization.commands_nm.size):
        command = trajectory_feedback_command(
            control_state,
            selected.optimization.states[:, index],
            float(selected.optimization.commands_nm[index]),
            selected.tvlqr[index],
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        control_state = measurement.copy() if swing_observer is None else swing_observer.update(command, measurement)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("adaptive-tvlqr")

        if capture_supervisor and (index + 1) * dt >= capture_supervisor_start_s:
            capture_streak = capture_streak + 1 if _capture_ready(selected.plant, control_state) else 0
            if capture_streak >= required_capture_steps:
                handoff_step = index + 1
                break

    if handoff_step is None:
        handoff_step = selected.optimization.commands_nm.size
    switch_time_s = handoff_step * dt
    total_steps = int(round(total_s / dt))

    if sensor is None:
        hold_gain = selected.upright_lqr
        hold_estimator = None
    else:
        hold_gain = upright_discrete_lqr_gain(selected.plant)
        hold_estimator = ExtendedKalmanObserver(selected.plant, control_state, sensor_noise)

    while len(states) < total_steps:
        hold_state = control_state if hold_estimator is None else hold_estimator.estimate
        command = lqr_command(
            hold_state,
            selected.optimization.target_state,
            hold_gain,
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        control_state = measurement.copy() if hold_estimator is None else hold_estimator.update(command, measurement)
        times.append((len(states) + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("supervised-lqr" if capture_supervisor else "adaptive-lqr")

    history = SimulationHistory(
        times_s=np.asarray(times),
        states=np.asarray(states),
        commands_nm=np.asarray(commands),
        modes=np.asarray(modes, dtype="U24"),
        switch_time_s=float(switch_time_s),
    )
    return AdaptiveSimulationResult(
        history=history,
        selected_model=selected_name,
        identification_errors=estimator.normalized_errors(),
        identification_steps=identify_steps,
    )
