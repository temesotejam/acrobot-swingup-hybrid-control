from __future__ import annotations

import math

import numpy as np

from acrobot_hybrid.adaptive import ModelBankEstimator, candidate_physics_models
from acrobot_hybrid.controllers import (
    lqr_command,
    numerical_discrete_linearization,
    trajectory_feedback_command,
    upright_discrete_lqr_gain,
    upright_lqr_gain,
)
from acrobot_hybrid.holdout import holdout_physics_models
from acrobot_hybrid.optimization import generate_energy_seed
from acrobot_hybrid.plant import AcrobotPlant, PhysicsConfig
from acrobot_hybrid.robustness import default_robustness_scenarios
from acrobot_hybrid.sensing import ExtendedKalmanObserver, NoisyStateSensor, SensorNoiseConfig


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
    target = np.array([math.pi, 0.0, 0.0, 0.0, 0.0])
    for gain in (upright_lqr_gain(plant), upright_discrete_lqr_gain(plant)):
        state = target.copy()
        state[0] += math.radians(0.1)
        initial_error = abs(state[0] - target[0])
        for _ in range(int(round(2.0 / plant.config.dt_s))):
            command = lqr_command(state, target, gain, plant.config.max_torque_nm)
            state = plant.step(state, command)
        assert abs(state[0] - target[0]) < initial_error
        assert plant.tip_height_m(state) > 1.99


def test_discrete_linearization_matches_step_dimensions() -> None:
    plant = AcrobotPlant(PhysicsConfig(), wrap_angles=False)
    state = np.array([0.2, -0.1, 0.3, -0.2, 0.05], dtype=np.float64)
    a, b = numerical_discrete_linearization(plant, state, 0.2)
    assert a.shape == (5, 5)
    assert b.shape == (5, 1)
    assert np.isfinite(a).all()
    assert np.isfinite(b).all()
    assert abs(b[4, 0]) > 0.1


def test_trajectory_feedback_has_full_physical_torque_authority() -> None:
    state = np.array([0.1, 0.0, 0.0, 0.0, 0.0])
    nominal = np.zeros(5)
    gain = np.array([[100.0, 0.0, 0.0, 0.0, 0.0]])
    command = trajectory_feedback_command(state, nominal, 0.98, gain, 1.0)
    assert command == -1.0


def test_robustness_scenario_groups_are_explicit() -> None:
    scenarios = default_robustness_scenarios(PhysicsConfig())
    groups = [item.group for item in scenarios]
    assert groups.count("initial-state") == 8
    assert groups.count("model-mismatch") == 8
    assert groups.count("stress") == 4


def test_model_bank_matches_model_mismatch_uncertainty_set() -> None:
    models = candidate_physics_models(PhysicsConfig())
    assert len(models) == 9
    assert "nominal" in models
    scenario_names = {item.name for item in default_robustness_scenarios(PhysicsConfig()) if item.group == "model-mismatch"}
    assert scenario_names == set(models) - {"nominal"}


def test_model_bank_estimator_selects_exact_transition_model() -> None:
    models = candidate_physics_models(PhysicsConfig())
    state = np.array([0.7, -0.6, 1.2, -0.8, 0.3], dtype=np.float64)
    commands = [0.8, -0.4, 0.2]
    for expected_name, config in models.items():
        actual = AcrobotPlant(config, wrap_angles=False)
        estimator = ModelBankEstimator(models)
        current = state.copy()
        for command in commands:
            next_state = actual.step(current, command)
            estimator.update(current, command, next_state)
            current = next_state
        assert estimator.selected_model() == expected_name


def test_holdout_models_are_not_stored_bank_entries() -> None:
    nominal = PhysicsConfig()
    bank = list(candidate_physics_models(nominal).values())
    holdouts = holdout_physics_models(nominal)
    assert len(holdouts) == 8
    assert all(item.physics not in bank for item in holdouts)


def test_noisy_sensor_is_reproducible_and_episode_biased() -> None:
    config = SensorNoiseConfig()
    sensor_a = NoisyStateSensor(config, seed=123)
    sensor_b = NoisyStateSensor(config, seed=123)
    state = np.zeros(5, dtype=np.float64)
    observations_a = [sensor_a.observe(state) for _ in range(3)]
    observations_b = [sensor_b.observe(state) for _ in range(3)]
    for left, right in zip(observations_a, observations_b, strict=True):
        assert np.allclose(left, right)
    assert np.linalg.norm(sensor_a.bias[:2]) > 0.0


def test_extended_kalman_observer_reduces_white_noise_at_upright() -> None:
    plant = AcrobotPlant(PhysicsConfig(), wrap_angles=False)
    config = SensorNoiseConfig(angle_bias_std_deg=0.0, gyro_bias_std_dps=0.0, torque_bias_std_nm=0.0)
    sensor = NoisyStateSensor(config, seed=7)
    target = np.array([math.pi, 0.0, 0.0, 0.0, 0.0])
    first = sensor.observe(target)
    observer = ExtendedKalmanObserver(plant, first, config)
    raw_errors = []
    filtered_errors = []
    for _ in range(100):
        measurement = sensor.observe(target)
        estimate = observer.update(0.0, measurement)
        raw_errors.append(abs(measurement[0] - target[0]))
        filtered_errors.append(abs(estimate[0] - target[0]))
    assert np.mean(filtered_errors[20:]) < np.mean(raw_errors[20:])
