from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from .plant import AcrobotPlant


def numerical_linearization(
    plant: AcrobotPlant,
    equilibrium: np.ndarray,
    command_nm: float = 0.0,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(equilibrium, dtype=np.float64)
    n = x0.size
    a = np.zeros((n, n), dtype=np.float64)
    b = np.zeros((n, 1), dtype=np.float64)
    for index in range(n):
        delta = np.zeros(n, dtype=np.float64)
        delta[index] = epsilon
        a[:, index] = (
            plant.derivative(x0 + delta, command_nm) - plant.derivative(x0 - delta, command_nm)
        ) / (2.0 * epsilon)
    b[:, 0] = (
        plant.derivative(x0, command_nm + epsilon) - plant.derivative(x0, command_nm - epsilon)
    ) / (2.0 * epsilon)
    return a, b


def numerical_discrete_linearization(
    plant: AcrobotPlant,
    state: np.ndarray,
    command_nm: float,
    state_epsilon: float = 1e-5,
    command_epsilon: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the exact RK4 one-step map used by the simulator."""
    x0 = np.asarray(state, dtype=np.float64)
    n = x0.size
    a = np.zeros((n, n), dtype=np.float64)
    for index in range(n):
        delta = np.zeros(n, dtype=np.float64)
        delta[index] = state_epsilon
        a[:, index] = (
            plant.step(x0 + delta, command_nm) - plant.step(x0 - delta, command_nm)
        ) / (2.0 * state_epsilon)
    b = (
        (plant.step(x0, command_nm + command_epsilon) - plant.step(x0, command_nm - command_epsilon))
        / (2.0 * command_epsilon)
    ).reshape(n, 1)
    return a, b


def upright_lqr_gain(plant: AcrobotPlant) -> np.ndarray:
    equilibrium = np.array([math.pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a, b = numerical_linearization(plant, equilibrium)
    q = np.diag([100.0, 50.0, 10.0, 10.0, 1.0])
    r = np.array([[1.0]], dtype=np.float64)
    p = solve_continuous_are(a, b, q, r)
    return np.linalg.solve(r, b.T @ p)


def upright_discrete_lqr_gain(
    plant: AcrobotPlant,
    q_diagonal: tuple[float, float, float, float, float] = (80.0, 40.0, 5.0, 5.0, 0.25),
    r_weight: float = 6.0,
) -> np.ndarray:
    """Discrete 50 Hz upright LQR tuned to avoid amplifying measurement noise.

    It uses the exact simulator step map and a larger input penalty than the
    historical continuous-time gain. The latter remains the noiseless baseline;
    this gain is used only by the noisy estimated-state hold controller.
    """
    equilibrium = np.array([math.pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a, b = numerical_discrete_linearization(plant, equilibrium, 0.0)
    q = np.diag(np.asarray(q_diagonal, dtype=np.float64))
    r = np.array([[float(r_weight)]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)


def tvlqr_gains(
    plant: AcrobotPlant,
    nominal_states: np.ndarray,
    nominal_commands_nm: np.ndarray,
    q_diagonal: tuple[float, float, float, float, float] = (2.0, 1.0, 0.5, 0.2, 0.05),
    r_weight: float = 5.0,
) -> np.ndarray:
    """Finite-horizon discrete TVLQR around the optimized swing-up trajectory."""
    states = np.asarray(nominal_states, dtype=np.float64)
    commands = np.asarray(nominal_commands_nm, dtype=np.float64).reshape(-1)
    if states.shape != (5, commands.size + 1):
        raise ValueError("nominal_states must have shape (5, len(commands)+1)")

    q = np.diag(np.asarray(q_diagonal, dtype=np.float64))
    r = np.array([[float(r_weight)]], dtype=np.float64)
    linearizations = [
        numerical_discrete_linearization(plant, states[:, index], float(commands[index]))
        for index in range(commands.size)
    ]

    terminal_a, terminal_b = numerical_discrete_linearization(plant, states[:, -1], 0.0)
    try:
        p = solve_discrete_are(terminal_a, terminal_b, q, r)
    except Exception:
        p = np.diag([500.0, 300.0, 50.0, 30.0, 5.0])

    gains = np.zeros((commands.size, 1, 5), dtype=np.float64)
    for index in range(commands.size - 1, -1, -1):
        a, b = linearizations[index]
        s = r + b.T @ p @ b
        gain = np.linalg.solve(s, b.T @ p @ a)
        gains[index] = gain
        p = q + a.T @ p @ (a - b @ gain)
        p = 0.5 * (p + p.T)
    return gains


def lqr_command(state: np.ndarray, target: np.ndarray, gain: np.ndarray, max_torque_nm: float) -> float:
    error = np.asarray(state, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    command = float(-(gain @ error)[0])
    return float(np.clip(command, -max_torque_nm, max_torque_nm))


def trajectory_feedback_command(
    state: np.ndarray,
    nominal_state: np.ndarray,
    nominal_command_nm: float,
    gain: np.ndarray,
    max_torque_nm: float,
) -> float:
    error = np.asarray(state, dtype=np.float64) - np.asarray(nominal_state, dtype=np.float64)
    correction = float((gain @ error)[0])
    return float(np.clip(float(nominal_command_nm) - correction, -max_torque_nm, max_torque_nm))


def energy_seed_command(
    plant: AcrobotPlant,
    state: np.ndarray,
    time_s: float,
    energy_gain: float = 0.20,
    kick_amplitude_nm: float = 1.0,
    kick_frequency_rad_s: float = 2.0,
    kick_duration_s: float = 4.0,
) -> float:
    energy_error = plant.upright_energy_j() - plant.mechanical_energy_j(state)
    excitation = kick_amplitude_nm * math.sin(kick_frequency_rad_s * time_s) if time_s < kick_duration_s else 0.0
    command = energy_gain * energy_error * float(state[3]) + excitation
    return float(np.clip(command, -plant.config.max_torque_nm, plant.config.max_torque_nm))
