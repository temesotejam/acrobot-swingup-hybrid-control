from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import lqr_command, upright_lqr_gain
from .optimization import OptimizationResult
from .plant import AcrobotPlant


@dataclass
class SimulationHistory:
    times_s: np.ndarray
    states: np.ndarray
    commands_nm: np.ndarray
    modes: np.ndarray
    switch_time_s: float


def simulate_optimized_hybrid(
    plant: AcrobotPlant,
    optimization: OptimizationResult,
    hold_s: float = 20.0,
) -> SimulationHistory:
    state = np.zeros(5, dtype=np.float64)
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
        modes.append("trajectory")

    switch_time_s = len(optimization.commands_nm) * dt
    gain = upright_lqr_gain(plant)
    hold_steps = int(round(hold_s / dt))
    for index in range(hold_steps):
        command = lqr_command(state, optimization.target_state, gain, plant.config.max_torque_nm)
        state = plant.step(state, command)
        times.append(switch_time_s + (index + 1) * dt)
        states.append(state.copy())
        commands.append(command)
        modes.append("lqr")

    return SimulationHistory(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        commands_nm=np.asarray(commands, dtype=np.float64),
        modes=np.asarray(modes, dtype="U16"),
        switch_time_s=float(switch_time_s),
    )
