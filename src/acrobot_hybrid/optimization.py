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
    nominal_torque_limit_nm: float


def generate_energy_seed(plant: AcrobotPlant, horizon_s: float = 21.0) -> tuple[np.ndarray, np.ndarray]:
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


def _configure_solver(opti: ca.Opti, max_iter: int = 1600) -> None:
    opti.solver(
        "ipopt",
        {"expand": True, "print_time": False},
        {"max_iter": max_iter, "print_level": 0, "sb": "yes", "tol": 1e-6, "acceptable_tol": 1e-4},
    )


def optimize_nominal_trajectory(
    plant: AcrobotPlant,
    horizon_s: float = 21.0,
    nominal_torque_limit_nm: float = 0.98,
) -> OptimizationResult:
    """Optimize a swing-up trajectory while reserving actuator feedback headroom."""
    seed_states, seed_commands = generate_energy_seed(plant, horizon_s=horizon_s)
    p = plant.config
    steps = seed_commands.size
    torque_limit = float(nominal_torque_limit_nm)
    if not 0.0 < torque_limit <= p.max_torque_nm:
        raise ValueError("nominal_torque_limit_nm must be within the physical torque bound")

    theta1_turn = round((seed_states[-1, 0] - math.pi) / (2.0 * math.pi))
    theta2_turn = round(seed_states[-1, 1] / (2.0 * math.pi))
    target = np.array([math.pi + 2.0 * math.pi * theta1_turn, 2.0 * math.pi * theta2_turn, 0.0, 0.0, 0.0], dtype=np.float64)

    opti = ca.Opti()
    x = opti.variable(5, steps + 1)
    u = opti.variable(1, steps)
    opti.subject_to(x[:, 0] == ca.DM.zeros(5, 1))
    for index in range(steps):
        opti.subject_to(x[:, index + 1] == _rk4(x[:, index], u[:, index], p))
    opti.subject_to(opti.bounded(-torque_limit, u, torque_limit))

    error = x[:, -1] - ca.DM(target)
    objective = 10000.0 * error[0] ** 2 + 5000.0 * error[1] ** 2 + 1500.0 * error[2] ** 2 + 800.0 * error[3] ** 2 + 100.0 * error[4] ** 2
    settle_start = int(0.85 * steps)
    for index in range(steps):
        objective += 0.001 * u[0, index] ** 2
        if index >= settle_start:
            settle_error = x[:, index] - ca.DM(target)
            objective += 0.20 * settle_error[0] ** 2 + 0.10 * settle_error[1] ** 2 + 0.05 * settle_error[2] ** 2 + 0.03 * settle_error[3] ** 2
    opti.minimize(objective)
    opti.set_initial(x, seed_states.T)
    opti.set_initial(u, np.clip(seed_commands, -torque_limit, torque_limit).reshape(1, -1))
    _configure_solver(opti)
    solution = opti.solve()
    return OptimizationResult(
        states=np.asarray(solution.value(x), dtype=np.float64),
        commands_nm=np.asarray(solution.value(u), dtype=np.float64).reshape(-1),
        target_state=target,
        objective=float(solution.value(objective)),
        solver_status=str(opti.stats().get("return_status", "unknown")),
        nominal_torque_limit_nm=torque_limit,
    )


def refine_trajectory_for_model(
    plant: AcrobotPlant,
    reference: OptimizationResult,
    nominal_torque_limit_nm: float = 0.99,
    terminal_angle_tolerance_deg: float = 0.5,
    terminal_velocity_tolerance_rad_s: float = 0.05,
    terminal_torque_tolerance_nm: float = 0.10,
) -> OptimizationResult:
    """Re-optimize a known-good swing-up for a nearby model."""
    p = plant.config
    steps = reference.commands_nm.size
    if reference.states.shape != (5, steps + 1):
        raise ValueError("reference trajectory shape is inconsistent")
    torque_limit = float(nominal_torque_limit_nm)
    target = np.asarray(reference.target_state, dtype=np.float64).copy()
    opti = ca.Opti()
    x = opti.variable(5, steps + 1)
    u = opti.variable(1, steps)
    opti.subject_to(x[:, 0] == ca.DM.zeros(5, 1))
    for index in range(steps):
        opti.subject_to(x[:, index + 1] == _rk4(x[:, index], u[:, index], p))
    opti.subject_to(opti.bounded(-torque_limit, u, torque_limit))

    terminal = x[:, -1] - ca.DM(target)
    angle_tol = math.radians(float(terminal_angle_tolerance_deg))
    velocity_tol = float(terminal_velocity_tolerance_rad_s)
    torque_tol = float(terminal_torque_tolerance_nm)
    opti.subject_to(opti.bounded(-angle_tol, terminal[0], angle_tol))
    opti.subject_to(opti.bounded(-angle_tol, terminal[1], angle_tol))
    opti.subject_to(opti.bounded(-velocity_tol, terminal[2], velocity_tol))
    opti.subject_to(opti.bounded(-velocity_tol, terminal[3], velocity_tol))
    opti.subject_to(opti.bounded(-torque_tol, terminal[4], torque_tol))

    objective = 1500.0 * terminal[0] ** 2 + 800.0 * terminal[1] ** 2 + 300.0 * terminal[2] ** 2 + 180.0 * terminal[3] ** 2 + 20.0 * terminal[4] ** 2
    state_weights = np.array([0.05, 0.03, 0.01, 0.006, 0.001], dtype=np.float64)
    for index in range(steps):
        state_error = x[:, index] - ca.DM(reference.states[:, index])
        command_error = u[0, index] - float(reference.commands_nm[index])
        objective += 0.01 * command_error**2
        for state_index, weight in enumerate(state_weights):
            objective += float(weight) * state_error[state_index] ** 2
    opti.minimize(objective)
    opti.set_initial(x, reference.states)
    opti.set_initial(u, np.clip(reference.commands_nm, -torque_limit, torque_limit).reshape(1, -1))
    _configure_solver(opti, max_iter=1200)
    solution = opti.solve()
    return OptimizationResult(
        states=np.asarray(solution.value(x), dtype=np.float64),
        commands_nm=np.asarray(solution.value(u), dtype=np.float64).reshape(-1),
        target_state=target,
        objective=float(solution.value(objective)),
        solver_status=str(opti.stats().get("return_status", "unknown")),
        nominal_torque_limit_nm=torque_limit,
    )


def replan_terminal_from_state(
    plant: AcrobotPlant,
    start_state: np.ndarray,
    reference: OptimizationResult,
    start_index: int,
    torque_limit_nm: float = 1.0,
    terminal_angle_tolerance_deg: float = 0.35,
    terminal_velocity_tolerance_rad_s: float = 0.035,
    terminal_torque_tolerance_nm: float = 0.08,
) -> OptimizationResult:
    """Short nonlinear MPC-style replan from the current estimate to upright.

    This is intended for the final few seconds of swing-up. The remaining
    selected trajectory is a warm start, but the first state is replaced by the
    latest filtered estimate and IPOPT is required to return to the local LQR
    basin under the selected/interpolated dynamics.
    """
    p = plant.config
    start_index = int(start_index)
    if not 0 <= start_index < reference.commands_nm.size:
        raise ValueError("start_index must lie inside the reference trajectory")
    ref_states = np.asarray(reference.states[:, start_index:], dtype=np.float64)
    ref_commands = np.asarray(reference.commands_nm[start_index:], dtype=np.float64)
    steps = ref_commands.size
    if steps < 10:
        raise ValueError("terminal replan needs at least 10 control steps")
    start = np.asarray(start_state, dtype=np.float64).copy()
    target = np.asarray(reference.target_state, dtype=np.float64).copy()
    torque_limit = min(float(torque_limit_nm), float(p.max_torque_nm))

    opti = ca.Opti()
    x = opti.variable(5, steps + 1)
    u = opti.variable(1, steps)
    opti.subject_to(x[:, 0] == ca.DM(start))
    for index in range(steps):
        opti.subject_to(x[:, index + 1] == _rk4(x[:, index], u[:, index], p))
    opti.subject_to(opti.bounded(-torque_limit, u, torque_limit))

    terminal = x[:, -1] - ca.DM(target)
    angle_tol = math.radians(float(terminal_angle_tolerance_deg))
    velocity_tol = float(terminal_velocity_tolerance_rad_s)
    torque_tol = float(terminal_torque_tolerance_nm)
    opti.subject_to(opti.bounded(-angle_tol, terminal[0], angle_tol))
    opti.subject_to(opti.bounded(-angle_tol, terminal[1], angle_tol))
    opti.subject_to(opti.bounded(-velocity_tol, terminal[2], velocity_tol))
    opti.subject_to(opti.bounded(-velocity_tol, terminal[3], velocity_tol))
    opti.subject_to(opti.bounded(-torque_tol, terminal[4], torque_tol))

    objective = 3000.0 * terminal[0] ** 2 + 1800.0 * terminal[1] ** 2 + 500.0 * terminal[2] ** 2 + 300.0 * terminal[3] ** 2 + 30.0 * terminal[4] ** 2
    for index in range(steps):
        state_error = x[:, index] - ca.DM(ref_states[:, index])
        command_error = u[0, index] - float(ref_commands[index])
        objective += 0.004 * command_error**2
        objective += 0.005 * state_error[0] ** 2 + 0.003 * state_error[1] ** 2 + 0.001 * state_error[2] ** 2 + 0.001 * state_error[3] ** 2
    opti.minimize(objective)

    # Warm-start with the remaining reference plus a linearly decaying state
    # offset so the initial guess satisfies the new current-state boundary.
    offset = start - ref_states[:, 0]
    guess_states = ref_states.copy()
    for index in range(steps + 1):
        fraction = 1.0 - index / steps
        guess_states[:, index] += fraction * offset
    opti.set_initial(x, guess_states)
    opti.set_initial(u, np.clip(ref_commands, -torque_limit, torque_limit).reshape(1, -1))
    _configure_solver(opti, max_iter=700)
    solution = opti.solve()
    return OptimizationResult(
        states=np.asarray(solution.value(x), dtype=np.float64),
        commands_nm=np.asarray(solution.value(u), dtype=np.float64).reshape(-1),
        target_state=target,
        objective=float(solution.value(objective)),
        solver_status=str(opti.stats().get("return_status", "unknown")),
        nominal_torque_limit_nm=torque_limit,
    )
