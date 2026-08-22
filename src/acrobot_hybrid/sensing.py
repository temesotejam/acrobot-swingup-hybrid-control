from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .plant import AcrobotPlant


@dataclass(frozen=True)
class SensorNoiseConfig:
    """State-measurement noise used by robustness tests."""

    angle_white_std_deg: float = 0.25
    angle_bias_std_deg: float = 1.0
    gyro_white_std_dps: float = 0.10
    gyro_bias_std_dps: float = 0.30
    torque_white_std_nm: float = 0.003
    torque_bias_std_nm: float = 0.003


class NoisyStateSensor:
    def __init__(self, config: SensorNoiseConfig, seed: int):
        self.config = config
        self.rng = np.random.default_rng(int(seed))
        self.bias = np.array(
            [
                math.radians(self.rng.normal(0.0, config.angle_bias_std_deg)),
                math.radians(self.rng.normal(0.0, config.angle_bias_std_deg)),
                math.radians(self.rng.normal(0.0, config.gyro_bias_std_dps)),
                math.radians(self.rng.normal(0.0, config.gyro_bias_std_dps)),
                self.rng.normal(0.0, config.torque_bias_std_nm),
            ],
            dtype=np.float64,
        )
        self.white_std = np.array(
            [
                math.radians(config.angle_white_std_deg),
                math.radians(config.angle_white_std_deg),
                math.radians(config.gyro_white_std_dps),
                math.radians(config.gyro_white_std_dps),
                config.torque_white_std_nm,
            ],
            dtype=np.float64,
        )

    def observe(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        return state + self.bias + self.rng.normal(0.0, self.white_std)


class ModelPredictiveObserver:
    """Small fixed-gain predict/correct filter retained for swing-up tracking."""

    def __init__(
        self,
        plant: AcrobotPlant,
        initial_measurement: np.ndarray,
        correction_gains: tuple[float, float, float, float, float] = (0.18, 0.18, 0.35, 0.35, 0.50),
    ):
        self.plant = plant
        self.estimate = np.asarray(initial_measurement, dtype=np.float64).copy()
        self.correction = np.asarray(correction_gains, dtype=np.float64)

    def set_plant(self, plant: AcrobotPlant) -> None:
        self.plant = plant

    def update(self, command_nm: float, measurement: np.ndarray) -> np.ndarray:
        predicted = self.plant.step(self.estimate, float(command_nm))
        residual = np.asarray(measurement, dtype=np.float64) - predicted
        self.estimate = predicted + self.correction * residual
        return self.estimate.copy()


def _step_jacobian(plant: AcrobotPlant, state: np.ndarray, command_nm: float, epsilon: float = 1e-5) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    jacobian = np.zeros((5, 5), dtype=np.float64)
    for index in range(5):
        delta = np.zeros(5, dtype=np.float64)
        delta[index] = epsilon
        jacobian[:, index] = (
            plant.step(state + delta, command_nm) - plant.step(state - delta, command_nm)
        ) / (2.0 * epsilon)
    return jacobian


class ExtendedKalmanObserver:
    """Nonlinear process-model EKF for the full measured Acrobot state."""

    def __init__(
        self,
        plant: AcrobotPlant,
        initial_state: np.ndarray,
        sensor_config: SensorNoiseConfig,
    ):
        self.plant = plant
        self.estimate = np.asarray(initial_state, dtype=np.float64).copy()
        angle_sigma = math.radians(sensor_config.angle_white_std_deg)
        gyro_sigma = math.radians(sensor_config.gyro_white_std_dps)
        torque_sigma = sensor_config.torque_white_std_nm
        self.r = np.diag(
            np.square([angle_sigma * 1.2, angle_sigma * 1.2, gyro_sigma * 1.3, gyro_sigma * 1.3, torque_sigma * 1.2])
        )
        self.q = np.diag([2e-8, 2e-8, 3e-6, 3e-6, 2e-5])
        self.covariance = self.r.copy()
        self.identity = np.eye(5, dtype=np.float64)

    def set_plant(self, plant: AcrobotPlant) -> None:
        self.plant = plant

    def reset(self, estimate: np.ndarray, covariance_scale: float = 1.0) -> None:
        self.estimate = np.asarray(estimate, dtype=np.float64).copy()
        self.covariance = self.r * float(covariance_scale)

    def update(self, command_nm: float, measurement: np.ndarray) -> np.ndarray:
        a = _step_jacobian(self.plant, self.estimate, float(command_nm))
        predicted = self.plant.step(self.estimate, float(command_nm))
        p_pred = a @ self.covariance @ a.T + self.q
        innovation_cov = p_pred + self.r
        kalman_gain = np.linalg.solve(innovation_cov.T, p_pred.T).T
        innovation = np.asarray(measurement, dtype=np.float64) - predicted
        self.estimate = predicted + kalman_gain @ innovation
        self.covariance = (self.identity - kalman_gain) @ p_pred
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return self.estimate.copy()
