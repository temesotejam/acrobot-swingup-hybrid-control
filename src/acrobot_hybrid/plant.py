from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: float = 9.8
    link_length_1_m: float = 1.0
    link_length_2_m: float = 1.0
    link_mass_1_kg: float = 1.0
    link_mass_2_kg: float = 1.0
    link_com_1_m: float = 0.5
    link_com_2_m: float = 0.5
    link_moi_1_kg_m2: float = 1.0
    link_moi_2_kg_m2: float = 1.0
    max_torque_nm: float = 1.0
    joint1_damping_nm_per_rad_s: float = 0.02
    joint2_damping_nm_per_rad_s: float = 0.02
    motor_time_constant_s: float = 0.05
    dt_s: float = 0.02
    max_velocity_1_rad_s: float = 4.0 * math.pi
    max_velocity_2_rad_s: float = 9.0 * math.pi

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class AcrobotPlant:
    """Continuous-torque Acrobot with first-order actuator dynamics.

    State is [theta1, theta2, dtheta1, dtheta2, actual_torque].
    theta1=0 points link 1 downward. theta2 is relative to link 1.
    Commanded torque is clipped to +/- max_torque_nm and applied at joint 2.
    """

    def __init__(self, config: PhysicsConfig | None = None, wrap_angles: bool = False):
        self.config = config or PhysicsConfig()
        self.wrap_angles = bool(wrap_angles)

    def derivative(self, state: np.ndarray, commanded_torque_nm: float) -> np.ndarray:
        p = self.config
        theta1, theta2, dtheta1, dtheta2, actual_torque = [float(v) for v in state]
        m1, m2 = p.link_mass_1_kg, p.link_mass_2_kg
        l1 = p.link_length_1_m
        lc1, lc2 = p.link_com_1_m, p.link_com_2_m
        i1, i2 = p.link_moi_1_kg_m2, p.link_moi_2_kg_m2
        g = p.gravity

        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * math.cos(theta2)) + i1 + i2
        d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(theta2)) + i2
        phi2 = m2 * lc2 * g * math.cos(theta1 + theta2 - math.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * math.sin(theta2)
            - 2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * math.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * math.cos(theta1 - math.pi / 2.0)
            + phi2
        )

        tau1 = -p.joint1_damping_nm_per_rad_s * dtheta1
        tau2 = actual_torque - p.joint2_damping_nm_per_rad_s * dtheta2
        denominator = m2 * lc2**2 + i2 - d2**2 / d1
        coriolis2 = m2 * l1 * lc2 * dtheta1**2 * math.sin(theta2)
        ddtheta2 = (tau2 - d2 / d1 * tau1 + d2 / d1 * phi1 - coriolis2 - phi2) / denominator
        ddtheta1 = (tau1 - d2 * ddtheta2 - phi1) / d1

        command = float(np.clip(commanded_torque_nm, -p.max_torque_nm, p.max_torque_nm))
        dtorque = (command - actual_torque) / max(p.motor_time_constant_s, 1e-9)
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, dtorque], dtype=np.float64)

    def step(self, state: np.ndarray, commanded_torque_nm: float) -> np.ndarray:
        p = self.config
        x = np.asarray(state, dtype=np.float64)
        h = p.dt_s
        k1 = self.derivative(x, commanded_torque_nm)
        k2 = self.derivative(x + 0.5 * h * k1, commanded_torque_nm)
        k3 = self.derivative(x + 0.5 * h * k2, commanded_torque_nm)
        k4 = self.derivative(x + h * k3, commanded_torque_nm)
        next_state = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        next_state[2] = float(np.clip(next_state[2], -p.max_velocity_1_rad_s, p.max_velocity_1_rad_s))
        next_state[3] = float(np.clip(next_state[3], -p.max_velocity_2_rad_s, p.max_velocity_2_rad_s))
        if self.wrap_angles:
            next_state[0] = wrap_angle(float(next_state[0]))
            next_state[1] = wrap_angle(float(next_state[1]))
        return next_state

    def tip_height_m(self, state: np.ndarray) -> float:
        theta1, theta2 = [float(v) for v in state[:2]]
        p = self.config
        return float(-p.link_length_1_m * math.cos(theta1) - p.link_length_2_m * math.cos(theta1 + theta2))

    def mechanical_energy_j(self, state: np.ndarray) -> float:
        theta1, theta2, dtheta1, dtheta2, _ = [float(v) for v in state]
        p = self.config
        m1, m2 = p.link_mass_1_kg, p.link_mass_2_kg
        l1 = p.link_length_1_m
        lc1, lc2 = p.link_com_1_m, p.link_com_2_m
        i1, i2 = p.link_moi_1_kg_m2, p.link_moi_2_kg_m2

        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * math.cos(theta2)) + i1 + i2
        d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(theta2)) + i2
        m22 = m2 * lc2**2 + i2
        kinetic = 0.5 * (d1 * dtheta1**2 + 2.0 * d2 * dtheta1 * dtheta2 + m22 * dtheta2**2)
        y1 = -lc1 * math.cos(theta1)
        y2 = -l1 * math.cos(theta1) - lc2 * math.cos(theta1 + theta2)
        potential = p.gravity * (m1 * y1 + m2 * y2)
        return float(kinetic + potential)

    def upright_energy_j(self) -> float:
        return self.mechanical_energy_j(np.array([math.pi, 0.0, 0.0, 0.0, 0.0]))
