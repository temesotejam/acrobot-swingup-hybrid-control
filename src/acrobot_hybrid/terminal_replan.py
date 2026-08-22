from __future__ import annotations

import numpy as np

from .adaptive import (
    AdaptiveLibraryEntry,
    AdaptiveSimulationResult,
    ModelBankEstimator,
    _interpolated_library_entry,
)
from .controllers import lqr_command, trajectory_feedback_command, tvlqr_gains, upright_discrete_lqr_gain
from .optimization import replan_terminal_from_state
from .plant import AcrobotPlant
from .sensing import ExtendedKalmanObserver, ModelPredictiveObserver, NoisyStateSensor, SensorNoiseConfig
from .simulation import SimulationHistory


def simulate_terminal_replan_hybrid(
    simulation_plant: AcrobotPlant,
    library: dict[str, AdaptiveLibraryEntry],
    identification_s: float = 1.0,
    replan_start_s: float = 18.0,
    total_s: float = 40.0,
    sensor_noise: SensorNoiseConfig | None = None,
    sensor_seed: int = 0,
    calibration_samples: int = 50,
) -> AdaptiveSimulationResult:
    """Adaptive TVLQR with one late nonlinear MPC-style terminal replan.

    The first identification window chooses a continuous model/trajectory for
    normal swing-up.  Dynamics evidence continues to accumulate while tracking.
    At `replan_start_s`, the model is estimated again from the whole observed
    history and IPOPT solves only the remaining trajectory to the upright LQR
    basin from the latest filtered state estimate.
    """
    if "nominal" not in library:
        raise ValueError("adaptive library must contain nominal")
    nominal = library["nominal"]
    dt = simulation_plant.config.dt_s
    estimator = ModelBankEstimator({name: entry.plant.config for name, entry in library.items()})
    identify_steps = int(round(identification_s / dt))
    replan_index = int(round(replan_start_s / dt))
    replan_index = max(identify_steps + 1, min(replan_index, nominal.optimization.commands_nm.size - 10))

    state = np.zeros(5, dtype=np.float64)
    sensor = None if sensor_noise is None else NoisyStateSensor(sensor_noise, sensor_seed)
    bias_estimate = np.zeros(5, dtype=np.float64)
    if sensor is not None and calibration_samples > 0:
        calibration = np.asarray([sensor.observe(state) for _ in range(int(calibration_samples))])
        bias_estimate = np.mean(calibration, axis=0) - state

    def observe(value: np.ndarray) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64).copy() if sensor is None else sensor.observe(value)
        return raw - bias_estimate

    measurement = observe(state)
    observer = None if sensor is None else ModelPredictiveObserver(nominal.plant, measurement)
    control_state = measurement.copy() if observer is None else observer.estimate.copy()
    times: list[float] = []
    states: list[np.ndarray] = []
    commands: list[float] = []
    modes: list[str] = []

    # Initial identification under the nominal controller.
    for index in range(identify_steps):
        command = trajectory_feedback_command(
            control_state,
            nominal.optimization.states[:, index],
            float(nominal.optimization.commands_nm[index]),
            nominal.tvlqr[index],
            simulation_plant.config.max_torque_nm,
        )
        before = measurement.copy()
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        estimator.update(before, command, measurement)
        control_state = measurement.copy() if observer is None else observer.update(command, measurement)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("identify")

    early_estimate = estimator.continuous_estimate(nominal.plant.config)
    selected = _interpolated_library_entry(library, early_estimate)
    if observer is not None:
        observer.set_plant(selected.plant)

    # Continue the selected trajectory while collecting much more dynamics data.
    for index in range(identify_steps, replan_index):
        command = trajectory_feedback_command(
            control_state,
            selected.optimization.states[:, index],
            float(selected.optimization.commands_nm[index]),
            selected.tvlqr[index],
            simulation_plant.config.max_torque_nm,
        )
        before = measurement.copy()
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        estimator.update(before, command, measurement)
        control_state = measurement.copy() if observer is None else observer.update(command, measurement)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("adaptive-tvlqr")

    # Re-identify from the full 18 s evidence and solve a short terminal NLP.
    late_estimate = estimator.continuous_estimate(nominal.plant.config)
    selected = _interpolated_library_entry(library, late_estimate)
    if observer is not None:
        observer.set_plant(selected.plant)
    replan = replan_terminal_from_state(
        selected.plant,
        control_state,
        selected.optimization,
        replan_index,
        torque_limit_nm=simulation_plant.config.max_torque_nm,
    )
    replan_gains = tvlqr_gains(selected.plant, replan.states, replan.commands_nm)

    for local_index, nominal_command in enumerate(replan.commands_nm):
        command = trajectory_feedback_command(
            control_state,
            replan.states[:, local_index],
            float(nominal_command),
            replan_gains[local_index],
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        control_state = measurement.copy() if observer is None else observer.update(command, measurement)
        global_index = replan_index + local_index
        times.append((global_index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("terminal-replan")

    switch_time_s = len(states) * dt
    total_steps = int(round(total_s / dt))
    if sensor is None:
        hold_estimator = None
        hold_gain = selected.upright_lqr
    else:
        hold_estimator = ExtendedKalmanObserver(selected.plant, control_state, sensor_noise)
        hold_gain = upright_discrete_lqr_gain(selected.plant)

    while len(states) < total_steps:
        hold_state = control_state if hold_estimator is None else hold_estimator.estimate
        command = lqr_command(
            hold_state,
            replan.target_state,
            hold_gain,
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        control_state = measurement.copy() if hold_estimator is None else hold_estimator.update(command, measurement)
        times.append((len(states) + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("replan-lqr")

    history = SimulationHistory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        commands_nm=np.asarray(commands, dtype=np.float64),
        modes=np.asarray(modes, dtype="U24"),
        switch_time_s=float(switch_time_s),
    )
    return AdaptiveSimulationResult(
        history=history,
        selected_model=f"{late_estimate.family} alpha={late_estimate.alpha:+.3f}",
        identification_errors=estimator.normalized_errors(),
        identification_steps=estimator.samples,
    )
