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


def build(results: Path, output: Path, run_url: str = "") -> None:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    robustness = summary["robustness_summary"]
    robustness_rows = summary["robustness_rows"]
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
        group_rows.append(
            "<tr>"
            f"<td>{html.escape(labels[group])}</td>"
            f"<td>{int(open_loop['final_stable_count'])}/{int(open_loop['count'])}</td>"
            f"<td><strong>{int(tvlqr['final_stable_count'])}/{int(tvlqr['count'])}</strong></td>"
            f"<td>{int(tvlqr['capture_count'])}/{int(tvlqr['count'])}</td>"
            f"<td>{_pct(tvlqr['mean_stable_ratio'])}</td>"
            "</tr>"
        )

    detail_rows = []
    for row in robustness_rows:
        if row["controller"] != "tvlqr":
            continue
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['scenario']))}</td>"
            f"<td>{html.escape(str(row['group']))}</td>"
            f"<td>{'yes' if row['capture'] else 'no'}</td>"
            f"<td>{_time(row['capture_time_s'])}</td>"
            f"<td>{'yes' if row['final_stable'] else 'no'}</td>"
            f"<td>{_pct(row['stable_ratio'])}</td>"
            f"<td>{float(row['rms_torque_nm']):.3f} N m</td>"
            "</tr>"
        )

    capture_time = metrics["capture_time_s"]
    page = f"""<!doctype html>
<html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Acrobot Hybrid Control</title><style>
:root{{font-family:system-ui,sans-serif;color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3}}main{{max-width:1180px;margin:auto;padding:28px 18px 64px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
video,img{{width:100%;border-radius:10px;background:#000}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #30363d;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#58a6ff}}.ok{{font-size:1.4rem;font-weight:700}}.scroll{{overflow:auto}}
</style></head><body><main>
<section class='card'><h1>Acrobot Swing-up Hybrid Control</h1><p>Energy-shaping seed → CasADi/IPOPT direct multiple shooting → <strong>TVLQR trajectory feedback</strong> → upright LQR で、真下からSwing-upと安定化を行います。</p><p>名目軌道は物理上限±1.00 N mのうち±{summary['nominal_trajectory_torque_limit_nm']:.2f} N mだけを使い、残りを閉ループ補正用のheadroomとして残します。</p><p class='ok'>Capture: {metrics['capture']} / Final stable: {metrics['final_stable']} / Initial-state robustness gate: {summary['robustness_gate']}</p><p><a href='{html.escape(run_url)}'>GitHub Actions run</a></p></section>
<section class='card'><h2>Nominal result</h2><table><tbody>
<tr><td>Solver</td><td>{html.escape(str(summary['solver_status']))}</td></tr>
<tr><td>TVLQR → upright LQR switch</td><td>{summary['switch_time_s']:.2f} s</td></tr>
<tr><td>Capture time</td><td>{capture_time:.2f} s</td></tr>
<tr><td>Stable dwell</td><td>{metrics['stable_ratio']*100:.1f}%</td></tr>
<tr><td>Final 2 s stable</td><td>{metrics['final_2s_stable_ratio']*100:.1f}%</td></tr>
<tr><td>Final tip height</td><td>{metrics['final_tip_height_m']:.6f} m</td></tr>
<tr><td>RMS torque</td><td>{metrics['rms_torque_nm']:.3f} N m</td></tr>
<tr><td>Nominal torque bound</td><td>±{summary['nominal_trajectory_torque_limit_nm']:.2f} N m</td></tr>
<tr><td>Feedback headroom</td><td>{summary['feedback_headroom_nm']:.2f} N m / side</td></tr>
</tbody></table></section>
<section class='card'><h2>Robustness: open-loop vs TVLQR</h2><p>初期状態8条件はCI gateです。モデル誤差と±5°は、今後MPCで改善すべき範囲を隠さず表示するstress diagnosticです。</p><div class='scroll'><table><thead><tr><th>Group</th><th>Open-loop Final stable</th><th>TVLQR Final stable</th><th>TVLQR Capture</th><th>TVLQR stable dwell</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div></section>
<section class='card'><h2>TVLQR scenario detail</h2><div class='scroll'><table><thead><tr><th>Scenario</th><th>Group</th><th>Capture</th><th>Capture time</th><th>Final stable</th><th>Stable dwell</th><th>RMS torque</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div></section>
<section class='card'><h2>Simulation</h2><video controls muted loop playsinline src='videos/hybrid.mp4'></video></section>
<section class='grid'><div class='card'><h2>Tip height</h2><img src='plots/height.png'></div><div class='card'><h2>Torque</h2><img src='plots/torque.png'></div></section>
<section class='card'><h2>States</h2><img src='plots/states.png'></section>
<section class='card'><h2>Interpretation</h2><p>Open-loop trajectory optimization established reachability. TVLQR now turns that trajectory into a local closed-loop funnel: the CI requires every ±2° / velocity-offset test to return to Final stable. Parameter mismatch is intentionally harder because the gains are designed only from the nominal model; that residual gap is the next target for nonlinear MPC / replanning.</p></section>
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
