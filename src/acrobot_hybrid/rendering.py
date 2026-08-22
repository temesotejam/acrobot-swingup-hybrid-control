from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .plant import AcrobotPlant
from .simulation import SimulationHistory


def _frame(plant: AcrobotPlant, state: np.ndarray, command: float, mode: str, time_s: float) -> np.ndarray:
    width = height = 480
    image = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    pivot = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    scale = 95.0
    theta1, theta2 = state[:2]
    p1 = pivot + np.array([math.sin(theta1), math.cos(theta1)]) * plant.config.link_length_1_m * scale
    p2 = p1 + np.array([math.sin(theta1 + theta2), math.cos(theta1 + theta2)]) * plant.config.link_length_2_m * scale
    target_y = pivot[1] - scale
    draw.line((30, target_y, width - 30, target_y), fill=(130, 130, 135), width=2)
    draw.line((*pivot, *p1), fill=(57, 106, 177), width=13)
    draw.line((*p1, *p2), fill=(205, 73, 62), width=13)
    for point in (pivot, p1, p2):
        x, y = point
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(30, 35, 42))
    draw.text((16, 16), f"mode: {mode}", fill=(30, 35, 42))
    draw.text((16, 38), f"t: {time_s:5.2f} s", fill=(30, 35, 42))
    draw.text((16, 60), f"tip: {plant.tip_height_m(state):+.2f} m", fill=(30, 35, 42))
    draw.text((16, 82), f"u: {command:+.2f} N m", fill=(30, 35, 42))
    return np.asarray(image, dtype=np.uint8)


def write_video(plant: AcrobotPlant, history: SimulationHistory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        _frame(plant, history.states[index], history.commands_nm[index], str(history.modes[index]), history.times_s[index])
        for index in range(0, len(history.times_s), 2)
    ]
    imageio.mimsave(path, frames, fps=25, macro_block_size=1)


def write_plots(plant: AcrobotPlant, history: SimulationHistory, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t = history.times_s
    x = history.states
    switch = history.switch_time_s

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, np.degrees(np.unwrap(x[:, 0])), label="theta1")
    axes[0].plot(t, np.degrees(np.unwrap(x[:, 1])), label="theta2")
    axes[0].axvline(switch, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Angle [deg]")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, x[:, 2], label="dtheta1")
    axes[1].plot(t, x[:, 3], label="dtheta2")
    axes[1].axvline(switch, linestyle="--", linewidth=1)
    axes[1].set_ylabel("Angular velocity [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "states.png", dpi=150)
    plt.close(fig)

    heights = np.asarray([plant.tip_height_m(state) for state in x])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, heights)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="capture height")
    ax.axhline(2.0, linestyle=":", linewidth=1, label="upright")
    ax.axvline(switch, linestyle="--", linewidth=1, label="LQR switch")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tip height [m]")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "height.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, history.commands_nm)
    ax.axvline(switch, linestyle="--", linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Commanded torque [N m]")
    ax.set_ylim(-1.1 * plant.config.max_torque_nm, 1.1 * plant.config.max_torque_nm)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "torque.png", dpi=150)
    plt.close(fig)
