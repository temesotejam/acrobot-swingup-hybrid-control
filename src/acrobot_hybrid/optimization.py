from __future__ import annotations

from dataclasses import dataclass
import math

import casadi as ca
import numpy as np

from .controllers import energy_seed_command
from .plant import AcrobotPlant, PhysicsConfig


@dataclass
class OptimizationResult:
    states: np.ndarray
    commands_nm: np.ndarray
    target_state: np.ndarray
    objective: float
    solver_status: str


def generate_energy_seed(plant: AcrobotPlant, horizon_s: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    steps = int(round(horizon_s / plant.config.dt_s))
    state = np.zeros(5, dtype=np.float64)
    states = [state.copy()]
    commands: list[float] = []
    for index in range(steps):
        time_s = index * plant.config.dt_s
        command = energy_seed_command(plant, state, time_s)
        commands.append(command)
        state = plant.step(state, command)
        states.append(state.copy())
    return np.asarray(states), np.asarray(commands)


def _symbolic_derivative(x, u, p: PhysicsConfig):
    theta1, theta2, dtheta1, dtheta2, actual_torque = [x[index] for index in range(5)]
    m1, m2 = p.link_mass_1_kg, p.link_mass_2_kg
    l1 = p.link_length_1_m
    lc1, lc2 = p.link_com_1_m, p.link_com_2_m
    i1, i2 = p.link_moi_1_kg_m2, p.link_moi_2_kg_m2
    g = p.gravity

    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * ca.cos(theta2)) + i1 + i2
    d2 = m2 * (lc2**2 + l1 * lc2 * ca.cos(theta2)) + i2
    phi2 = m2 * lc2 * g * ca.cos(theta1 + theta2 - ca.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * ca.sin(theta2)
        - 2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * ca.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * ca.cos(theta1 - ca.pi / 2.0)
        + phi2
    )
    tau1 = -p.joint1_damping_nm_per_rad_s * dtheta1
    tau2 = actual_torque - p.joint2_damping_nm_per_rad_s * dtheta2
    denominator = m2 * lc2**2 + i2 - d2**2 / d1
    coriolis2 = m2 * l1 * lc2 * dtheta1**2 * ca.sin(theta2)
    ddtheta2 = (tau2 - d2 / d1 * tau1 + d2 / d1 * phi1 - coriolis2 - phi2) / denominator
    ddtheta1 = (tau1 - d2 * ddtheta2 - phi1) / d1
    dtorque = (u - actual_torque) / p.motor_time_constant_s
    return ca.vertcat(dtheta1, dtheta2, ddtheta1, ddtheta2, dtorque)


def _rk4(x, u, p: PhysicsConfig):
    h = p.dt_s
    k1 = _symbolic_derivative(x, u, p)
    k2 = _symbolic_derivative(x + 0.5 * h * k1, u, p)
    k3 = _symbolic_derivative(x + 0.5 * h * k2, u, p)
    k4 = _symbolic_derivative(x + h * k3, u, p)
    return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def optimize_nominal_trajectory(
    plant: AcrobotPlant,
    horizon_s: float = 20.0,
) -> OptimizationResult:
    """Direct multiple-shooting optimization seeded by deterministic energy shaping.

    The energy-shaping seed gets the underactuated mechanism close to the
    upright energy manifold. IPOPT then adjusts the bounded torque history so
    the terminal state is close enough to the upright equilibrium for the local
    LQR controller to take over.
    """
    seed_states, seed_commands = generate_energy_seed(plant, horizon_s=horizon_s)
    p = plant.config
    steps = seed_commands.size

    theta1_turn = round((seed_states[-1, 0] - math.pi) / (2.0 * math.pi))
    theta2_turn = round(seed_states[-1, 1] / (2.0 * math.pi))
    target = np.array(
        [math.pi + 2.0 * math.pi * theta1_turn, 2.0 * math.pi * theta2_turn, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    opti = ca.Opti()
    x = opti.variable(5, steps + 1)
    u = opti.variable(1, steps)
    opti.subject_to(x[:, 0] == ca.DM.zeros(5, 1))
    for index in range(steps):
        opti.subject_to(x[:, index + 1] == _rk4(x[:, index], u[:, index], p))
    opti.subject_to(opti.bounded(-p.max_torque_nm, u, p.max_torque_nm))

    error = x[:, -1] - ca.DM(target)
    objective = (
        2000.0 * error[0] ** 2
        + 1000.0 * error[1] ** 2
        + 300.0 * error[2] ** 2
        + 150.0 * error[3] ** 2
        + 20.0 * error[4] ** 2
    )
    settle_start = int(0.90 * steps)
    for index in range(steps):
        objective += 0.002 * u[0, index] ** 2
        if index >= settle_start:
            e1 = x[0, index] - target[0]
            e2 = x[1, index] - target[1]
            objective += 0.10 * e1**2 + 0.05 * e2**2 + 0.02 * x[2, index] ** 2 + 0.01 * x[3, index] ** 2
    opti.minimize(objective)

    opti.set_initial(x, seed_states.T)
    opti.set_initial(u, seed_commands.reshape(1, -1))
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {"max_iter": 1200, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4},
    )
    solution = opti.solve()
    return OptimizationResult(
        states=np.asarray(solution.value(x), dtype=np.float64),
        commands_nm=np.asarray(solution.value(u), dtype=np.float64).reshape(-1),
        target_state=target,
        objective=float(solution.value(objective)),
        solver_status=str(opti.stats().get("return_status", "unknown")),
    )
