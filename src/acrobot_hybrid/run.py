from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .adaptive import build_adaptive_library
from .controllers import tvlqr_gains, upright_lqr_gain
from .evaluation import evaluate_history
from .holdout import evaluate_holdout_robustness
from .optimization import optimize_nominal_trajectory
from .plant import AcrobotPlant, PhysicsConfig, wrap_angle
from .rendering import write_plots, write_video
from .robustness import evaluate_robustness
from .sensing import SensorNoiseConfig
from .simulation import simulate_optimized_hybrid, simulate_tvlqr_hybrid


def _write_csv(path: Path, times: np.ndarray, states: np.ndarray, commands: np.ndarray, modes: np.ndarray, plant: AcrobotPlant) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "theta1_rad", "theta2_rad", "dtheta1_rad_s", "dtheta2_rad_s", "actual_torque_nm", "command_nm", "tip_height_m", "mode"])
        for time_s, state, command, mode in zip(times, states, commands, modes, strict=True):
            writer.writerow([time_s, *state.tolist(), command, plant.tip_height_m(state), mode])


def _write_robustness_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scenario", "group", "controller", "selected_model", "capture", "capture_time_s", "final_stable",
        "stable_ratio", "final_2s_stable_ratio", "final_tip_height_m", "rms_torque_nm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_holdout_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "scenario", "condition", "sensor_seed", "selected_model", "capture", "capture_time_s",
        "final_stable", "stable_ratio", "final_2s_stable_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _count_text(group: dict) -> str:
    return f"{int(group['final_stable_count'])}/{int(group['count'])}"


def _capture_text(group: dict) -> str:
    return f"{int(group['capture_count'])}/{int(group['count'])}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize and simulate the adaptive feedback-stabilized Acrobot swing-up.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--swingup-seconds", type=float, default=21.0)
    parser.add_argument("--hold-seconds", type=float, default=19.0)
    parser.add_argument("--nominal-torque-limit", type=float, default=0.98)
    parser.add_argument("--candidate-torque-limit", type=float, default=0.99)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    plots = output / "plots"
    videos = output / "videos"
    output.mkdir(parents=True, exist_ok=True)

    physics = PhysicsConfig()
    plant = AcrobotPlant(physics, wrap_angles=False)
    optimization = optimize_nominal_trajectory(
        plant,
        horizon_s=args.swingup_seconds,
        nominal_torque_limit_nm=args.nominal_torque_limit,
    )
    gains = tvlqr_gains(plant, optimization.states, optimization.commands_nm)
    adaptive_library = build_adaptive_library(
        plant,
        optimization,
        candidate_torque_limit_nm=args.candidate_torque_limit,
    )

    history = simulate_tvlqr_hybrid(
        plant,
        plant,
        optimization,
        gains=gains,
        hold_s=args.hold_seconds,
    )
    metrics = evaluate_history(plant, history.times_s, history.states, history.commands_nm)

    open_loop_history = simulate_optimized_hybrid(plant, optimization, hold_s=args.hold_seconds)
    open_loop_metrics = evaluate_history(
        plant,
        open_loop_history.times_s,
        open_loop_history.states,
        open_loop_history.commands_nm,
    )
    robustness_rows, robustness_summary = evaluate_robustness(
        plant,
        optimization,
        hold_s=args.hold_seconds,
        gains=gains,
        adaptive_library=adaptive_library,
    )

    sensor_noise = SensorNoiseConfig()
    holdout_rows, holdout_summary = evaluate_holdout_robustness(
        physics,
        adaptive_library,
        identification_s=0.5,
        noisy_seeds=(11, 22, 33),
        sensor_noise=sensor_noise,
    )

    terminal = optimization.states[:, -1]
    target = optimization.target_state
    optimizer_terminal = {
        "theta1_error_deg": math.degrees(wrap_angle(float(terminal[0] - target[0]))),
        "theta2_error_deg": math.degrees(wrap_angle(float(terminal[1] - target[1]))),
        "dtheta1_rad_s": float(terminal[2]),
        "dtheta2_rad_s": float(terminal[3]),
        "actual_torque_nm": float(terminal[4]),
    }
    initial_tvlqr = robustness_summary["initial-state"]["tvlqr"]
    initial_open = robustness_summary["initial-state"]["open-loop"]
    robustness_gate = (
        int(initial_tvlqr["final_stable_count"]) == int(initial_tvlqr["count"])
        and int(initial_tvlqr["final_stable_count"]) > int(initial_open["final_stable_count"])
    )

    model_tvlqr = robustness_summary["model-mismatch"]["tvlqr"]
    model_adaptive = robustness_summary["model-mismatch"]["adaptive"]
    model_rows = [
        row for row in robustness_rows
        if row["group"] == "model-mismatch" and row["controller"] == "adaptive"
    ]
    identification_correct = sum(row["selected_model"] == row["scenario"] for row in model_rows)
    adaptive_gate = (
        int(model_adaptive["final_stable_count"]) >= 6
        and int(model_adaptive["final_stable_count"]) >= int(model_tvlqr["final_stable_count"]) + 2
        and int(model_adaptive["capture_count"]) >= 7
        and identification_correct >= 7
    )

    clean_holdout = holdout_summary["clean-holdout"]
    noisy_holdout = holdout_summary["noisy-holdout"]
    holdout_gate = (
        int(clean_holdout["final_stable_count"]) >= 6
        and int(clean_holdout["capture_count"]) >= 7
        and int(noisy_holdout["final_stable_count"]) >= 18
        and int(noisy_holdout["capture_count"]) >= 20
    )

    library_summary: dict[str, dict] = {}
    for name, entry in adaptive_library.items():
        terminal_state = entry.optimization.states[:, -1]
        entry_target = entry.optimization.target_state
        library_summary[name] = {
            "solver_status": entry.optimization.solver_status,
            "objective": float(entry.optimization.objective),
            "torque_limit_nm": float(entry.optimization.nominal_torque_limit_nm),
            "terminal_theta1_error_deg": math.degrees(wrap_angle(float(terminal_state[0] - entry_target[0]))),
            "terminal_theta2_error_deg": math.degrees(wrap_angle(float(terminal_state[1] - entry_target[1]))),
            "terminal_dtheta1_rad_s": float(terminal_state[2]),
            "terminal_dtheta2_rad_s": float(terminal_state[3]),
        }

    summary = {
        "controller": "direct-multiple-shooting + TVLQR + multiple-model adaptive replanning + upright-LQR",
        "solver": "CasADi/IPOPT",
        "solver_status": optimization.solver_status,
        "objective": optimization.objective,
        "physics": physics.to_dict(),
        "swingup_seconds": args.swingup_seconds,
        "hold_seconds": args.hold_seconds,
        "switch_time_s": history.switch_time_s,
        "physical_torque_limit_nm": physics.max_torque_nm,
        "nominal_trajectory_torque_limit_nm": optimization.nominal_torque_limit_nm,
        "adaptive_candidate_torque_limit_nm": args.candidate_torque_limit,
        "feedback_headroom_nm": physics.max_torque_nm - optimization.nominal_torque_limit_nm,
        "optimizer_target_state": target.tolist(),
        "optimizer_terminal": optimizer_terminal,
        "upright_lqr_gain": upright_lqr_gain(plant).tolist(),
        "tvlqr_shape": list(gains.shape),
        "adaptive_library": library_summary,
        "adaptive_identification_correct": int(identification_correct),
        "adaptive_identification_total": int(len(model_rows)),
        "sensor_noise": sensor_noise.__dict__,
        "holdout_summary": holdout_summary,
        "holdout_rows": _json_safe(holdout_rows),
        "metrics": metrics.to_dict(),
        "open_loop_nominal_metrics": open_loop_metrics.to_dict(),
        "robustness_summary": robustness_summary,
        "robustness_rows": _json_safe(robustness_rows),
        "robustness_gate": bool(robustness_gate),
        "adaptive_model_mismatch_gate": bool(adaptive_gate),
        "holdout_noise_gate": bool(holdout_gate),
        "success": bool(metrics.capture and metrics.final_stable and robustness_gate and adaptive_gate and holdout_gate),
    }

    np.savez_compressed(
        output / "nominal_trajectory.npz",
        optimized_states=optimization.states,
        optimized_commands_nm=optimization.commands_nm,
        target_state=optimization.target_state,
        tvlqr_gains=gains,
        times_s=history.times_s,
        states=history.states,
        commands_nm=history.commands_nm,
        modes=history.modes,
    )
    np.save(output / "tvlqr_gains.npy", gains)
    _write_csv(output / "trajectory.csv", history.times_s, history.states, history.commands_nm, history.modes, plant)
    _write_robustness_csv(output / "robustness.csv", robustness_rows)
    _write_holdout_csv(output / "holdout.csv", holdout_rows)
    (output / "robustness.json").write_text(
        json.dumps(_json_safe({"summary": robustness_summary, "rows": robustness_rows}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "holdout.json").write_text(
        json.dumps(_json_safe({"summary": holdout_summary, "rows": holdout_rows, "sensor_noise": sensor_noise.__dict__}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "adaptive-library.json").write_text(
        json.dumps(_json_safe(library_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_plots(plant, history, plots)
    write_video(plant, history, videos / "hybrid.mp4")
    (output / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    capture_time = metrics.capture_time_s
    capture_text = "-" if math.isnan(capture_time) else f"{capture_time:.2f} s"
    initial_tv = robustness_summary["initial-state"]["tvlqr"]
    initial_ad = robustness_summary["initial-state"]["adaptive"]
    initial_ol = robustness_summary["initial-state"]["open-loop"]
    model_tv = robustness_summary["model-mismatch"]["tvlqr"]
    model_ad = robustness_summary["model-mismatch"]["adaptive"]
    model_ol = robustness_summary["model-mismatch"]["open-loop"]
    stress_tv = robustness_summary["stress"]["tvlqr"]
    stress_ad = robustness_summary["stress"]["adaptive"]
    stress_ol = robustness_summary["stress"]["open-loop"]
    markdown = f"""# Adaptive Hybrid Acrobot result

- controller: `direct multiple shooting + TVLQR + multiple-model adaptive replanning + upright LQR`
- solver: `CasADi/IPOPT` (`{optimization.solver_status}`)
- swing-up horizon: `{args.swingup_seconds:.1f} s`
- upright hold: `{args.hold_seconds:.1f} s`
- physical torque limit: `+/-{physics.max_torque_nm:.2f} N m`
- nominal trajectory torque limit: `+/-{optimization.nominal_torque_limit_nm:.2f} N m`
- candidate replan torque limit: `+/-{args.candidate_torque_limit:.2f} N m`
- Capture: **{metrics.capture}**
- Capture time: **{capture_text}**
- Final stable: **{metrics.final_stable}**
- Stable dwell: **{metrics.stable_ratio * 100:.1f}%**
- Final 2 s stable: **{metrics.final_2s_stable_ratio * 100:.1f}%**
- Final tip height: **{metrics.final_tip_height_m:.6f} m**
- RMS commanded torque: **{metrics.rms_torque_nm:.3f} N m**

## Robustness: open-loop vs TVLQR vs adaptive replanning

| Perturbation group | Open-loop final stable | TVLQR final stable | Adaptive final stable | Adaptive Capture |
|---|---:|---:|---:|---:|
| initial state: +/-2 deg / velocity offsets | {_count_text(initial_ol)} | {_count_text(initial_tv)} | **{_count_text(initial_ad)}** | {_capture_text(initial_ad)} |
| model mismatch | {_count_text(model_ol)} | {_count_text(model_tv)} | **{_count_text(model_ad)}** | {_capture_text(model_ad)} |
| +/-5 deg stress | {_count_text(stress_ol)} | {_count_text(stress_tv)} | **{_count_text(stress_ad)}** | {_capture_text(stress_ad)} |

## Hold-out generalization

| Condition | Final stable | Capture |
|---|---:|---:|
| bank-interior parameters, clean | **{_count_text(clean_holdout)}** | {_capture_text(clean_holdout)} |
| same parameters + sensor noise/bias, 3 seeds each | **{_count_text(noisy_holdout)}** | {_capture_text(noisy_holdout)} |

The hold-out values are not entries in the controller bank. Noisy tests use 0.25 deg angle white noise, 1.0 deg episode angle bias, 0.10 deg/s gyro white noise and 0.30 deg/s gyro bias throughout identification and feedback control.

`success=true` requires nominal performance, the original initial-state and endpoint-model gates, plus at least 6/8 clean hold-out Final stable and 18/24 noisy hold-out Final stable.
"""
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if not summary["success"]:
        raise SystemExit("Adaptive hybrid controller did not meet hold-out + noisy generalization criteria")


if __name__ == "__main__":
    main()
