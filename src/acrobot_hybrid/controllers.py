from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve_continuous_are

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


def upright_lqr_gain(plant: AcrobotPlant) -> np.ndarray:
    equilibrium = np.array([math.pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a, b = numerical_linearization(plant, equilibrium)
    q = np.diag([100.0, 50.0, 10.0, 10.0, 1.0])
    r = np.array([[1.0]], dtype=np.float64)
    p = solve_continuous_are(a, b, q, r)
    return np.linalg.solve(r, b.T @ p)


def lqr_command(state: np.ndarray, target: np.ndarray, gain: np.ndarray, max_torque_nm: float) -> float:
    error = np.asarray(state, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    command = float(-(gain @ error)[0])
    return float(np.clip(command, -max_torque_nm, max_torque_nm))


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
