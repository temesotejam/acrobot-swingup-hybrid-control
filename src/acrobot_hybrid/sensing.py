from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .plant import AcrobotPlant


@dataclass(frozen=True)
class SensorNoiseConfig:
    """State-measurement noise used by robustness tests.

    Defaults intentionally match the noise scale used in the companion RL
    benchmark for joint angles and gyros. Biases are sampled once per episode;
    white noise is sampled at every observation.
    """

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
    """Lightweight predict/correct filter for noisy five-state feedback.

    The model prediction carries fast swing-up motion without introducing the
    phase lag of a plain low-pass filter. Measurement correction suppresses
    white noise while still allowing fixed sensor bias and model mismatch to be
    visible to the controller/estimator robustness test.
    """

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
