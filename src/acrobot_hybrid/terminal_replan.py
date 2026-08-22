from __future__ import annotations

import casadi as ca
import numpy as np

from .adaptive import (
    AdaptiveLibraryEntry,
    AdaptiveSimulationResult,
    ModelBankEstimator,
    _interpolated_library_entry,
)
from .controllers import lqr_command, trajectory_feedback_command, tvlqr_gains, upright_discrete_lqr_gain
from .optimization import OptimizationResult, _configure_solver, _rk4
from .plant import AcrobotPlant
from .sensing import ExtendedKalmanObserver, ModelPredictiveObserver, NoisyStateSensor, SensorNoiseConfig
from .simulation import SimulationHistory


def _soft_terminal_replan(
    plant: AcrobotPlant,
    start_state: np.ndarray,
    reference: OptimizationResult,
    start_index: int,
) -> OptimizationResult:
    """Feasible short-horizon nonlinear replan with a strong soft terminal cost."""
    p = plant.config
    start_index = int(start_index)
    ref_states = np.asarray(reference.states[:, start_index:], dtype=np.float64)
    ref_commands = np.asarray(reference.commands_nm[start_index:], dtype=np.float64)
    steps = ref_commands.size
    if steps < 10:
        raise ValueError("terminal replan needs at least 10 steps")
    start = np.asarray(start_state, dtype=np.float64).copy()
    target = np.asarray(reference.target_state, dtype=np.float64).copy()

    opti = ca.Opti()
    x = opti.variable(5, steps + 1)
    u = opti.variable(1, steps)
    opti.subject_to(x[:, 0] == ca.DM(start))
    for index in range(steps):
        opti.subject_to(x[:, index + 1] == _rk4(x[:, index], u[:, index], p))
    opti.subject_to(opti.bounded(-p.max_torque_nm, u, p.max_torque_nm))

    terminal = x[:, -1] - ca.DM(target)
    objective = (
        120000.0 * terminal[0] ** 2
        + 70000.0 * terminal[1] ** 2
        + 18000.0 * terminal[2] ** 2
        + 10000.0 * terminal[3] ** 2
        + 1200.0 * terminal[4] ** 2
    )
    settle_start = max(0, steps - int(round(1.0 / p.dt_s)))
    for index in range(steps):
        command_error = u[0, index] - float(ref_commands[index])
        objective += 0.002 * command_error**2
        if index >= settle_start:
            settle = x[:, index] - ca.DM(target)
            objective += 80.0 * settle[0] ** 2 + 45.0 * settle[1] ** 2 + 14.0 * settle[2] ** 2 + 8.0 * settle[3] ** 2
    opti.minimize(objective)

    offset = start - ref_states[:, 0]
    guess_states = ref_states.copy()
    for index in range(steps + 1):
        guess_states[:, index] += (1.0 - index / steps) * offset
    opti.set_initial(x, guess_states)
    opti.set_initial(u, np.clip(ref_commands, -p.max_torque_nm, p.max_torque_nm).reshape(1, -1))
    _configure_solver(opti, max_iter=800)
    solution = opti.solve()
    return OptimizationResult(
        states=np.asarray(solution.value(x), dtype=np.float64),
        commands_nm=np.asarray(solution.value(u), dtype=np.float64).reshape(-1),
        target_state=target,
        objective=float(solution.value(objective)),
        solver_status=str(opti.stats().get("return_status", "unknown")),
        nominal_torque_limit_nm=p.max_torque_nm,
    )


def _reference_tail(reference: OptimizationResult, start_index: int) -> OptimizationResult:
    return OptimizationResult(
        states=np.asarray(reference.states[:, start_index:], dtype=np.float64).copy(),
        commands_nm=np.asarray(reference.commands_nm[start_index:], dtype=np.float64).copy(),
        target_state=np.asarray(reference.target_state, dtype=np.float64).copy(),
        objective=float("nan"),
        solver_status="fallback-reference-tail",
        nominal_torque_limit_nm=float(reference.nominal_torque_limit_nm),
    )


def simulate_terminal_replan_hybrid(
    simulation_plant: AcrobotPlant,
    library: dict[str, AdaptiveLibraryEntry],
    identification_s: float = 1.0,
    replan_start_s: float = 17.0,
    total_s: float = 40.0,
    sensor_noise: SensorNoiseConfig | None = None,
    sensor_seed: int = 0,
    calibration_samples: int = 50,
) -> AdaptiveSimulationResult:
    """Adaptive TVLQR with late re-identification and one nonlinear MPC replan."""
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

    late_estimate = estimator.continuous_estimate(nominal.plant.config)
    selected = _interpolated_library_entry(library, late_estimate)
    if observer is not None:
        observer.set_plant(selected.plant)

    try:
        replan = _soft_terminal_replan(selected.plant, control_state, selected.optimization, replan_index)
        replan_gains = tvlqr_gains(selected.plant, replan.states, replan.commands_nm)
        replan_mode = "terminal-replan"
    except RuntimeError:
        replan = _reference_tail(selected.optimization, replan_index)
        replan_gains = selected.tvlqr[replan_index:].copy()
        replan_mode = "replan-fallback"

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
        modes.append(replan_mode)

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
        command = lqr_command(hold_state, replan.target_state, hold_gain, simulation_plant.config.max_torque_nm)
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
