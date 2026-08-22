from __future__ import annotations

import numpy as np

from .adaptive import (
    AdaptiveLibraryEntry,
    AdaptiveSimulationResult,
    ModelBankEstimator,
    _interpolated_library_entry,
)
from .controllers import lqr_command, trajectory_feedback_command, tvlqr_gains, upright_discrete_lqr_gain
from .plant import AcrobotPlant
from .sensing import ExtendedKalmanObserver, ModelPredictiveObserver, NoisyStateSensor, SensorNoiseConfig
from .simulation import SimulationHistory
from .terminal_replan import _reference_tail, _soft_terminal_replan, _upright_lqr_ready


def simulate_receding_recovery_hybrid(
    simulation_plant: AcrobotPlant,
    library: dict[str, AdaptiveLibraryEntry],
    identification_s: float = 1.0,
    replan_start_s: float = 18.0,
    recovery_start_s: float = 19.5,
    total_s: float = 40.0,
    sensor_noise: SensorNoiseConfig | None = None,
    sensor_seed: int = 0,
    calibration_samples: int = 50,
    capture_supervisor_dwell_s: float = 0.20,
    recovery_extension_s: float = 2.0,
    mpc_apply_s: float = 0.75,
    max_recovery_cycles: int = 4,
) -> AdaptiveSimulationResult:
    """Noisy holdout controller with receding nonlinear basin recovery.

    The uncertainty family/model selected from the long 0--replan window is
    frozen during terminal recovery.  Run #27 showed that short-window family
    re-identification is less reliable under the benchmark sensor noise.

    Once recovery starts, only a short prefix of each basin-constrained plan is
    executed.  The optimizer is then solved again from the latest filtered
    state.  The local upright LQR is entered only after the measured/filtered
    state remains inside the empirically verified basin for the requested
    dwell; exhausting the MPC attempts never silently falls through to LQR.
    """
    if "nominal" not in library:
        raise ValueError("adaptive library must contain nominal")

    nominal = library["nominal"]
    dt = simulation_plant.config.dt_s
    model_configs = {name: entry.plant.config for name, entry in library.items()}
    estimator = ModelBankEstimator(model_configs)

    identify_steps = int(round(float(identification_s) / dt))
    replan_index = int(round(float(replan_start_s) / dt))
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

    def record(command: float, mode: str) -> None:
        times.append((len(states) + 1) * dt)
        states.append(state.copy())
        commands.append(float(command))
        modes.append(mode)

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
        record(command, "identify")

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
        record(command, "adaptive-tvlqr")

    # Freeze the long-window model for all terminal recovery.  This is more
    # repeatable than allowing a noisy 0.5--1.5 s terminal window to switch the
    # uncertainty family.
    selected_estimate = estimator.continuous_estimate(nominal.plant.config)
    selected = _interpolated_library_entry(library, selected_estimate)
    if observer is not None:
        observer.set_plant(selected.plant)

    lqr_gain = selected.upright_lqr if sensor is None else upright_discrete_lqr_gain(selected.plant)
    required_ready_steps = max(1, int(round(float(capture_supervisor_dwell_s) / dt)))
    ready_streak = 0
    handoff_ready = False

    try:
        first_plan = _soft_terminal_replan(selected.plant, control_state, selected.optimization, replan_index)
        first_gains = tvlqr_gains(selected.plant, first_plan.states, first_plan.commands_nm)
        first_mode = "terminal-replan"
    except RuntimeError:
        first_plan = _reference_tail(selected.optimization, replan_index)
        first_gains = selected.tvlqr[replan_index:].copy()
        first_mode = "replan-fallback"

    for local_index, nominal_command in enumerate(first_plan.commands_nm):
        command = trajectory_feedback_command(
            control_state,
            first_plan.states[:, local_index],
            float(nominal_command),
            first_gains[local_index],
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        measurement = observe(state)
        control_state = measurement.copy() if observer is None else observer.update(command, measurement)
        record(command, first_mode)

        if _upright_lqr_ready(selected.plant, control_state, first_plan.target_state, lqr_gain):
            ready_streak += 1
        else:
            ready_streak = 0
        if ready_streak >= required_ready_steps:
            handoff_ready = True
            break
        if len(states) * dt >= float(recovery_start_s):
            break

    last_plan = first_plan
    last_gains = first_gains
    last_applied = 0

    for cycle in range(int(max_recovery_cycles)):
        if handoff_ready:
            break

        recovery_index = min(len(states), selected.optimization.commands_nm.size - 10)
        try:
            plan = _soft_terminal_replan(
                selected.plant,
                control_state,
                selected.optimization,
                recovery_index,
                extension_s=recovery_extension_s,
                basin_gain=lqr_gain,
                basin_dwell_s=capture_supervisor_dwell_s,
            )
            plan_mode = f"mpc-basin-{cycle + 1}"
        except RuntimeError:
            try:
                plan = _soft_terminal_replan(
                    selected.plant,
                    control_state,
                    selected.optimization,
                    recovery_index,
                    extension_s=recovery_extension_s,
                )
                plan_mode = f"mpc-soft-{cycle + 1}"
            except RuntimeError:
                plan = _reference_tail(selected.optimization, recovery_index)
                plan_mode = f"mpc-fallback-{cycle + 1}"

        gains = tvlqr_gains(selected.plant, plan.states, plan.commands_nm)
        apply_steps = min(
            plan.commands_nm.size,
            max(required_ready_steps, int(round(float(mpc_apply_s) / dt))),
        )
        applied = 0
        for local_index in range(apply_steps):
            command = trajectory_feedback_command(
                control_state,
                plan.states[:, local_index],
                float(plan.commands_nm[local_index]),
                gains[local_index],
                simulation_plant.config.max_torque_nm,
            )
            state = simulation_plant.step(state, command)
            measurement = observe(state)
            control_state = measurement.copy() if observer is None else observer.update(command, measurement)
            record(command, plan_mode)
            applied = local_index + 1

            if _upright_lqr_ready(selected.plant, control_state, plan.target_state, lqr_gain):
                ready_streak += 1
            else:
                ready_streak = 0
            if ready_streak >= required_ready_steps:
                handoff_ready = True
                break

        last_plan = plan
        last_gains = gains
        last_applied = applied

    # If the cycle budget expires, continue the final nonlinear trajectory
    # instead of violating the supervisor and handing an out-of-basin state to
    # LQR.  This makes failure explicit while still giving the last recovery
    # plan a chance to enter the verified set.
    if not handoff_ready and last_plan.commands_nm.size > last_applied:
        for local_index in range(last_applied, last_plan.commands_nm.size):
            command = trajectory_feedback_command(
                control_state,
                last_plan.states[:, local_index],
                float(last_plan.commands_nm[local_index]),
                last_gains[local_index],
                simulation_plant.config.max_torque_nm,
            )
            state = simulation_plant.step(state, command)
            measurement = observe(state)
            control_state = measurement.copy() if observer is None else observer.update(command, measurement)
            record(command, "mpc-final-tail")

            if _upright_lqr_ready(selected.plant, control_state, last_plan.target_state, lqr_gain):
                ready_streak += 1
            else:
                ready_streak = 0
            if ready_streak >= required_ready_steps:
                handoff_ready = True
                break

    switch_time_s = len(states) * dt
    total_steps = int(round(float(total_s) / dt))

    if handoff_ready:
        hold_estimator = None
        if sensor is not None:
            hold_estimator = ExtendedKalmanObserver(selected.plant, control_state, sensor_noise)
        while len(states) < total_steps:
            hold_state = control_state if hold_estimator is None else hold_estimator.estimate
            command = lqr_command(
                hold_state,
                last_plan.target_state,
                lqr_gain,
                simulation_plant.config.max_torque_nm,
            )
            state = simulation_plant.step(state, command)
            measurement = observe(state)
            control_state = measurement.copy() if hold_estimator is None else hold_estimator.update(command, measurement)
            record(command, "verified-lqr")
    else:
        # Never claim a local-LQR handoff when the verified set was not reached.
        # Zero torque makes the exhausted-recovery failure deterministic and
        # keeps this branch diagnostic rather than hiding it behind saturated
        # local feedback.
        while len(states) < total_steps:
            command = 0.0
            state = simulation_plant.step(state, command)
            measurement = observe(state)
            control_state = measurement.copy() if observer is None else observer.update(command, measurement)
            record(command, "recovery-exhausted")

    history = SimulationHistory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        commands_nm=np.asarray(commands, dtype=np.float64),
        modes=np.asarray(modes, dtype="U24"),
        switch_time_s=float(switch_time_s),
    )
    return AdaptiveSimulationResult(
        history=history,
        selected_model=f"{selected_estimate.family} alpha={selected_estimate.alpha:+.3f}",
        identification_errors=estimator.normalized_errors(),
        identification_steps=estimator.samples,
    )
