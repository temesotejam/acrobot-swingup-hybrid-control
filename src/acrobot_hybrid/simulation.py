from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import lqr_command, trajectory_feedback_command, tvlqr_gains, upright_lqr_gain
from .optimization import OptimizationResult
from .plant import AcrobotPlant


@dataclass
class SimulationHistory:
    times_s: np.ndarray
    states: np.ndarray
    commands_nm: np.ndarray
    modes: np.ndarray
    switch_time_s: float


def _finish_with_lqr(
    controller_plant: AcrobotPlant,
    simulation_plant: AcrobotPlant,
    optimization: OptimizationResult,
    state: np.ndarray,
    hold_s: float,
    times: list[float],
    states: list[np.ndarray],
    commands: list[float],
    modes: list[str],
) -> None:
    switch_time_s = len(optimization.commands_nm) * simulation_plant.config.dt_s
    gain = upright_lqr_gain(controller_plant)
    hold_steps = int(round(hold_s / simulation_plant.config.dt_s))
    for index in range(hold_steps):
        command = lqr_command(
            state,
            optimization.target_state,
            gain,
            simulation_plant.config.max_torque_nm,
        )
        state[:] = simulation_plant.step(state, command)
        times.append(switch_time_s + (index + 1) * simulation_plant.config.dt_s)
        states.append(state.copy())
        commands.append(command)
        modes.append("lqr")


def simulate_optimized_hybrid(
    plant: AcrobotPlant,
    optimization: OptimizationResult,
    hold_s: float = 19.0,
    initial_state: np.ndarray | None = None,
) -> SimulationHistory:
    """Open-loop optimized trajectory followed by local LQR.

    This is retained as a robustness comparison baseline.  The nominal initial
    state is exactly downward unless an explicit perturbation is supplied.
    """
    state = np.zeros(5, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    times: list[float] = []
    states: list[np.ndarray] = []
    commands: list[float] = []
    modes: list[str] = []
    dt = plant.config.dt_s

    for index, command in enumerate(optimization.commands_nm):
        state = plant.step(state, float(command))
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(float(command))
        modes.append("open-loop")

    _finish_with_lqr(plant, plant, optimization, state, hold_s, times, states, commands, modes)
    return SimulationHistory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        commands_nm=np.asarray(commands, dtype=np.float64),
        modes=np.asarray(modes, dtype="U16"),
        switch_time_s=float(len(optimization.commands_nm) * dt),
    )


def simulate_tvlqr_hybrid(
    controller_plant: AcrobotPlant,
    simulation_plant: AcrobotPlant,
    optimization: OptimizationResult,
    gains: np.ndarray | None = None,
    hold_s: float = 19.0,
    initial_state: np.ndarray | None = None,
) -> SimulationHistory:
    """Track the optimized trajectory with TVLQR, then hand off to upright LQR.

    `controller_plant` is the nominal model used to design gains.  A different
    `simulation_plant` can be supplied to quantify parameter mismatch without
    quietly redesigning the controller for the perturbed plant.
    """
    if abs(controller_plant.config.dt_s - simulation_plant.config.dt_s) > 1e-12:
        raise ValueError("controller and simulation plants must share dt_s")
    if gains is None:
        gains = tvlqr_gains(controller_plant, optimization.states, optimization.commands_nm)

    state = np.zeros(5, dtype=np.float64) if initial_state is None else np.asarray(initial_state, dtype=np.float64).copy()
    times: list[float] = []
    states: list[np.ndarray] = []
    commands: list[float] = []
    modes: list[str] = []
    dt = simulation_plant.config.dt_s

    for index, nominal_command in enumerate(optimization.commands_nm):
        command = trajectory_feedback_command(
            state,
            optimization.states[:, index],
            float(nominal_command),
            gains[index],
            simulation_plant.config.max_torque_nm,
        )
        state = simulation_plant.step(state, command)
        times.append((index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("tvlqr")

    _finish_with_lqr(
        controller_plant,
        simulation_plant,
        optimization,
        state,
        hold_s,
        times,
        states,
        commands,
        modes,
    )
    return SimulationHistory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        commands_nm=np.asarray(commands, dtype=np.float64),
        modes=np.asarray(modes, dtype="U16"),
        switch_time_s=float(len(optimization.commands_nm) * dt),
    )
