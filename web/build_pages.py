from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path


def _pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _time(value) -> str:
    if value is None:
        return "-"
    number = float(value)
    return "-" if math.isnan(number) else f"{number:.2f} s"


def _summary_count(entry: dict, field: str) -> int:
    return int(entry.get(field, 0))


def build(results: Path, output: Path, run_url: str = "") -> None:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    robustness = summary["robustness_summary"]
    robustness_rows = summary["robustness_rows"]
    library = summary.get("adaptive_library", {})
    holdout_summary = summary.get("holdout_summary", {})
    noise = summary.get("sensor_noise", {})

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(results / "plots", output / "plots")
    shutil.copytree(results / "videos", output / "videos")

    group_rows = []
    labels = {
        "initial-state": "Initial state (+/-2 deg / velocity)",
        "model-mismatch": "Model mismatch",
        "stress": "+/-5 deg stress",
    }
    for group in ["initial-state", "model-mismatch", "stress"]:
        open_loop = robustness[group]["open-loop"]
        tvlqr = robustness[group]["tvlqr"]
        adaptive = robustness[group]["adaptive"]
        group_rows.append(
            "<tr>"
            f"<td>{html.escape(labels[group])}</td>"
            f"<td>{int(open_loop['final_stable_count'])}/{int(open_loop['count'])}</td>"
            f"<td>{int(tvlqr['final_stable_count'])}/{int(tvlqr['count'])}</td>"
            f"<td><strong>{int(adaptive['final_stable_count'])}/{int(adaptive['count'])}</strong></td>"
            f"<td>{int(adaptive['capture_count'])}/{int(adaptive['count'])}</td>"
            f"<td>{_pct(adaptive['mean_stable_ratio'])}</td>"
            "</tr>"
        )

    holdout_rows = []
    holdout_labels = {
        "clean-holdout": "Bank-interior holdout, clean",
        "noisy-holdout": "Same holdouts + sensor noise/bias",
    }
    for key in ["clean-holdout", "noisy-holdout"]:
        entry = holdout_summary.get(key)
        if not entry:
            continue
        holdout_rows.append(
            "<tr>"
            f"<td>{html.escape(holdout_labels[key])}</td>"
            f"<td>{_summary_count(entry, 'capture_count')}/{_summary_count(entry, 'count')}</td>"
            f"<td><strong>{_summary_count(entry, 'final_stable_count')}/{_summary_count(entry, 'count')}</strong></td>"
            f"<td>{_pct(entry.get('mean_stable_ratio', 0.0))}</td>"
            "</tr>"
        )

    detail_rows = []
    for row in robustness_rows:
        if row["controller"] != "adaptive":
            continue
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['scenario']))}</td>"
            f"<td>{html.escape(str(row['group']))}</td>"
            f"<td>{html.escape(str(row.get('selected_model') or '-'))}</td>"
            f"<td>{'yes' if row['capture'] else 'no'}</td>"
            f"<td>{_time(row['capture_time_s'])}</td>"
            f"<td>{'yes' if row['final_stable'] else 'no'}</td>"
            f"<td>{_pct(row['stable_ratio'])}</td>"
            f"<td>{float(row['rms_torque_nm']):.3f} N m</td>"
            "</tr>"
        )

    library_rows = []
    for name, entry in library.items():
        library_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(str(entry['solver_status']))}</td>"
            f"<td>±{float(entry['torque_limit_nm']):.2f} N m</td>"
            f"<td>{float(entry['terminal_theta1_error_deg']):.3f}°</td>"
            f"<td>{float(entry['terminal_theta2_error_deg']):.3f}°</td>"
            f"<td>{float(entry['terminal_dtheta1_rad_s']):.4f}</td>"
            f"<td>{float(entry['terminal_dtheta2_rad_s']):.4f}</td>"
            "</tr>"
        )

    success = bool(summary.get("success", False))
    holdout_gate = bool(summary.get("holdout_noise_gate", False))
    status_class = "pass" if success else "fail"
    status_text = "PASS" if success else "BENCHMARK GATE NOT MET"
    capture_time = metrics["capture_time_s"]
    run_link = (
        f"<a href='{html.escape(run_url)}'>GitHub Actions run</a>"
        if run_url
        else "GitHub Actions run unavailable"
    )
    noise_text = ""
    if noise:
        noise_text = (
            f"Angle white {float(noise.get('angle_white_std_deg', 0.0)):.2f}°, "
            f"angle bias {float(noise.get('angle_bias_std_deg', 0.0)):.2f}°, "
            f"gyro white {float(noise.get('gyro_white_std_dps', 0.0)):.2f}°/s, "
            f"gyro bias {float(noise.get('gyro_bias_std_dps', 0.0)):.2f}°/s"
        )

    page = f"""<!doctype html>
<html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Acrobot Hybrid Control</title><style>
:root{{font-family:system-ui,sans-serif;color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3}}main{{max-width:1220px;margin:auto;padding:28px 18px 64px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
video,img{{width:100%;border-radius:10px;background:#000}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #30363d;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#58a6ff}}.headline{{font-size:1.15rem;font-weight:700}}.scroll{{overflow:auto}}
.status{{display:inline-block;padding:7px 11px;border-radius:999px;font-weight:800}}.pass{{background:#123b25;color:#7ee787}}.fail{{background:#4a1f24;color:#ff7b72}}
.note{{color:#8b949e}}
</style></head><body><main>
<section class='card'><h1>Acrobot Swing-up Hybrid Control</h1>
<p><span class='status {status_class}'>{status_text}</span></p>
<p>Pages publication is independent from the benchmark gate: failed robustness runs are published intentionally so the current failure mode remains inspectable.</p>
<p>Energy-shaping seed → CasADi/IPOPT trajectory → TVLQR → adaptive model selection / replanning → receding nonlinear recovery → verified-basin upright LQR.</p>
<p class='headline'>Nominal Capture: {metrics['capture']} / Nominal Final stable: {metrics['final_stable']} / Holdout + noise gate: {holdout_gate}</p>
<p>{run_link}</p></section>

<section class='card'><h2>Nominal result</h2><table><tbody>
<tr><td>Solver</td><td>{html.escape(str(summary['solver_status']))}</td></tr>
<tr><td>TVLQR → upright LQR switch</td><td>{summary['switch_time_s']:.2f} s</td></tr>
<tr><td>Capture time</td><td>{_time(capture_time)}</td></tr>
<tr><td>Stable dwell</td><td>{metrics['stable_ratio']*100:.1f}%</td></tr>
<tr><td>Final 2 s stable</td><td>{metrics['final_2s_stable_ratio']*100:.1f}%</td></tr>
<tr><td>Final tip height</td><td>{metrics['final_tip_height_m']:.6f} m</td></tr>
<tr><td>RMS torque</td><td>{metrics['rms_torque_nm']:.3f} N m</td></tr>
<tr><td>Adaptive model identification</td><td>{summary['adaptive_identification_correct']}/{summary['adaptive_identification_total']}</td></tr>
</tbody></table></section>

<section class='card'><h2>Hold-out generalization</h2>
<p class='note'>This section is published even when the CI performance gate fails.</p>
<div class='scroll'><table><thead><tr><th>Condition</th><th>Capture</th><th>Final stable</th><th>Mean stable dwell</th></tr></thead>
<tbody>{''.join(holdout_rows) if holdout_rows else "<tr><td colspan='4'>No holdout summary available</td></tr>"}</tbody></table></div>
<p class='note'>{html.escape(noise_text)}</p></section>

<section class='card'><h2>Robustness: open-loop vs TVLQR vs Adaptive replanning</h2><div class='scroll'><table><thead><tr><th>Group</th><th>Open-loop Final stable</th><th>TVLQR Final stable</th><th>Adaptive Final stable</th><th>Adaptive Capture</th><th>Adaptive stable dwell</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div></section>

<section class='card'><h2>Adaptive scenario detail</h2><div class='scroll'><table><thead><tr><th>Scenario</th><th>Group</th><th>Selected model</th><th>Capture</th><th>Capture time</th><th>Final stable</th><th>Stable dwell</th><th>RMS torque</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div></section>

<section class='card'><h2>Adaptive trajectory library</h2><p>Each stored uncertainty endpoint is re-optimized from the nominal solution and used for interpolation / replanning.</p><div class='scroll'><table><thead><tr><th>Model</th><th>Solver</th><th>Torque bound</th><th>θ1 terminal</th><th>θ2 terminal</th><th>dθ1</th><th>dθ2</th></tr></thead><tbody>{''.join(library_rows)}</tbody></table></div></section>

<section class='card'><h2>Simulation</h2><video controls muted loop playsinline src='videos/hybrid.mp4'></video></section>
<section class='grid'><div class='card'><h2>Tip height</h2><img src='plots/height.png'></div><div class='card'><h2>Torque</h2><img src='plots/torque.png'></div></section>
<section class='card'><h2>States</h2><img src='plots/states.png'></section>

<section class='card'><h2>Current interpretation</h2>
<p>The nominal controller and the original initial-state / endpoint-model robustness tests remain strong. The current unresolved gate is noisy bank-interior holdout stabilization: the system can usually reach Capture, but it does not yet maintain the verified upright region through the final hold. This page intentionally exposes that failed result instead of suppressing deployment.</p>
</section>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()
    build(args.results, args.output, args.run_url)


if __name__ == "__main__":
    main()
