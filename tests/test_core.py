from __future__ import annotations

import math

import numpy as np

from acrobot_hybrid.controllers import lqr_command, upright_lqr_gain
from acrobot_hybrid.optimization import generate_energy_seed
from acrobot_hybrid.plant import AcrobotPlant, PhysicsConfig


def test_geometry_and_energy_gap_match_benchmark() -> None:
    plant = AcrobotPlant(PhysicsConfig(), wrap_angles=False)
    down = np.zeros(5)
    up = np.array([math.pi, 0.0, 0.0, 0.0, 0.0])
    assert abs(plant.tip_height_m(down) + 2.0) < 1e-12
    assert abs(plant.tip_height_m(up) - 2.0) < 1e-12
    assert abs((plant.mechanical_energy_j(up) - plant.mechanical_energy_j(down)) - 39.2) < 1e-9


def test_energy_seed_reaches_upright_energy_manifold() -> None:
    plant = AcrobotPlant(PhysicsConfig(), wrap_angles=False)
    states, commands = generate_energy_seed(plant, horizon_s=20.0)
    assert states.shape == (1001, 5)
    assert commands.shape == (1000,)
    assert np.max(np.abs(commands)) <= 1.0 + 1e-12
    assert plant.tip_height_m(states[-1]) > 1.8
    assert abs(plant.mechanical_energy_j(states[-1]) - plant.upright_energy_j()) < 1.0


def test_local_lqr_stabilizes_small_upright_perturbation() -> None:
    plant = AcrobotPlant(PhysicsConfig(), wrap_angles=False)
    gain = upright_lqr_gain(plant)
    target = np.array([math.pi, 0.0, 0.0, 0.0, 0.0])
    state = target.copy()
    state[0] += math.radians(0.1)
    initial_error = abs(state[0] - target[0])
    for _ in range(int(round(2.0 / plant.config.dt_s))):
        command = lqr_command(state, target, gain, plant.config.max_torque_nm)
        state = plant.step(state, command)
    assert abs(state[0] - target[0]) < initial_error
    assert plant.tip_height_m(state) > 1.99
