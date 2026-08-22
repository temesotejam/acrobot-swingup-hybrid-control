from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SensorNoiseConfig:
    """State-measurement noise used by robustness tests.

    Defaults intentionally match the noise scale used in the companion RL
    benchmark for joint angles and gyros.  Biases are sampled once per episode;
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
