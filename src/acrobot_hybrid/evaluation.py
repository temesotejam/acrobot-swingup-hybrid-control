from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .plant import AcrobotPlant, wrap_angle


@dataclass
class HybridMetrics:
    capture: bool
    capture_time_s: float
    final_stable: bool
    stable_ratio: float
    final_2s_stable_ratio: float
    max_tip_height_m: float
    final_tip_height_m: float
    final_theta1_error_deg: float
    final_theta2_error_deg: float
    final_dtheta1_rad_s: float
    final_dtheta2_rad_s: float
    rms_torque_nm: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def evaluate_history(
    plant: AcrobotPlant,
    times_s: np.ndarray,
    states: np.ndarray,
    commands_nm: np.ndarray,
) -> HybridMetrics:
    heights = np.asarray([plant.tip_height_m(state) for state in states], dtype=np.float64)
    stable = (heights >= 1.0) & (np.abs(states[:, 2]) <= 1.0) & (np.abs(states[:, 3]) <= 1.5)
    capture_steps = max(1, int(round(0.5 / plant.config.dt_s)))
    run = 0
    capture_time = math.nan
    for index, flag in enumerate(stable):
        run = run + 1 if bool(flag) else 0
        if math.isnan(capture_time) and run >= capture_steps:
            capture_time = float(times_s[index])
            break

    final_steps = max(1, int(round(2.0 / plant.config.dt_s)))
    final_ratio = float(np.mean(stable[-final_steps:]))
    final = states[-1]
    return HybridMetrics(
        capture=not math.isnan(capture_time),
        capture_time_s=capture_time,
        final_stable=final_ratio >= 0.80,
        stable_ratio=float(np.mean(stable)),
        final_2s_stable_ratio=final_ratio,
        max_tip_height_m=float(np.max(heights)),
        final_tip_height_m=float(heights[-1]),
        final_theta1_error_deg=float(math.degrees(wrap_angle(float(final[0]) - math.pi))),
        final_theta2_error_deg=float(math.degrees(wrap_angle(float(final[1])))),
        final_dtheta1_rad_s=float(final[2]),
        final_dtheta2_rad_s=float(final[3]),
        rms_torque_nm=float(np.sqrt(np.mean(np.square(commands_nm)))),
    )
