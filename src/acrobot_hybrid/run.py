from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .controllers import upright_lqr_gain
from .evaluation import evaluate_history
from .optimization import optimize_nominal_trajectory
from .plant import AcrobotPlant, PhysicsConfig, wrap_angle
from .rendering import write_plots, write_video
from .simulation import simulate_optimized_hybrid


def _write_csv(path: Path, times: np.ndarray, states: np.ndarray, commands: np.ndarray, modes: np.ndarray, plant: AcrobotPlant) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "theta1_rad", "theta2_rad", "dtheta1_rad_s", "dtheta2_rad_s", "actual_torque_nm", "command_nm", "tip_height_m", "mode"])
        for time_s, state, command, mode in zip(times, states, commands, modes, strict=True):
            writer.writerow([time_s, *state.tolist(), command, plant.tip_height_m(state), mode])


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize and simulate the hybrid Acrobot controller.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--swingup-seconds", type=float, default=20.0)
    parser.add_argument("--hold-seconds", type=float, default=20.0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    plots = output / "plots"
    videos = output / "videos"
    output.mkdir(parents=True, exist_ok=True)

    physics = PhysicsConfig()
    plant = AcrobotPlant(physics, wrap_angles=False)
    optimization = optimize_nominal_trajectory(plant, horizon_s=args.swingup_seconds)
    history = simulate_optimized_hybrid(plant, optimization, hold_s=args.hold_seconds)
    metrics = evaluate_history(plant, history.times_s, history.states, history.commands_nm)

    terminal = optimization.states[:, -1]
    target = optimization.target_state
    optimizer_terminal = {
        "theta1_error_deg": math.degrees(wrap_angle(float(terminal[0] - target[0]))),
        "theta2_error_deg": math.degrees(wrap_angle(float(terminal[1] - target[1]))),
        "dtheta1_rad_s": float(terminal[2]),
        "dtheta2_rad_s": float(terminal[3]),
        "actual_torque_nm": float(terminal[4]),
    }
    summary = {
        "controller": "direct-multiple-shooting + upright-LQR",
        "solver": "CasADi/IPOPT",
        "solver_status": optimization.solver_status,
        "objective": optimization.objective,
        "physics": physics.to_dict(),
        "swingup_seconds": args.swingup_seconds,
        "hold_seconds": args.hold_seconds,
        "switch_time_s": history.switch_time_s,
        "optimizer_target_state": target.tolist(),
        "optimizer_terminal": optimizer_terminal,
        "lqr_gain": upright_lqr_gain(plant).tolist(),
        "metrics": metrics.to_dict(),
        "success": bool(metrics.capture and metrics.final_stable),
    }

    np.savez_compressed(
        output / "nominal_trajectory.npz",
        optimized_states=optimization.states,
        optimized_commands_nm=optimization.commands_nm,
        target_state=optimization.target_state,
        times_s=history.times_s,
        states=history.states,
        commands_nm=history.commands_nm,
        modes=history.modes,
    )
    _write_csv(output / "trajectory.csv", history.times_s, history.states, history.commands_nm, history.modes, plant)
    write_plots(plant, history, plots)
    write_video(plant, history, videos / "hybrid.mp4")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    capture_time = metrics.capture_time_s
    capture_text = "-" if math.isnan(capture_time) else f"{capture_time:.2f} s"
    markdown = f"""# Hybrid Acrobot result

- controller: `direct multiple shooting + LQR`
- solver: `CasADi/IPOPT` (`{optimization.solver_status}`)
- swing-up optimization horizon: `{args.swingup_seconds:.1f} s`
- LQR hold: `{args.hold_seconds:.1f} s`
- Capture: **{metrics.capture}**
- Capture time: **{capture_text}**
- Final stable: **{metrics.final_stable}**
- Stable dwell: **{metrics.stable_ratio * 100:.1f}%**
- Final 2 s stable: **{metrics.final_2s_stable_ratio * 100:.1f}%**
- Final tip height: **{metrics.final_tip_height_m:.6f} m**
- Final theta1 error: **{metrics.final_theta1_error_deg:.6f} deg**
- Final theta2 error: **{metrics.final_theta2_error_deg:.6f} deg**
- RMS commanded torque: **{metrics.rms_torque_nm:.3f} N m**

## Optimizer terminal before LQR

- theta1 error: `{optimizer_terminal['theta1_error_deg']:.4f} deg`
- theta2 error: `{optimizer_terminal['theta2_error_deg']:.4f} deg`
- dtheta1: `{optimizer_terminal['dtheta1_rad_s']:.5f} rad/s`
- dtheta2: `{optimizer_terminal['dtheta2_rad_s']:.5f} rad/s`

`success=true` requires both Capture and Final stable on the full 40 s hybrid simulation.
"""
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if not summary["success"]:
        raise SystemExit("Hybrid controller did not meet Capture + Final stable criteria")


if __name__ == "__main__":
    main()
