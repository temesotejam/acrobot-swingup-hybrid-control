from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path


def build(results: Path, output: Path, run_url: str = "") -> None:
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(results / "plots", output / "plots")
    shutil.copytree(results / "videos", output / "videos")

    capture_time = metrics["capture_time_s"]
    page = f"""<!doctype html>
<html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Acrobot Hybrid Control</title><style>
:root{{font-family:system-ui,sans-serif;color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3}}main{{max-width:1100px;margin:auto;padding:28px 18px 64px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:20px;margin:16px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
video,img{{width:100%;border-radius:10px;background:#000}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #30363d;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#58a6ff}}.ok{{font-size:1.4rem;font-weight:700}}
</style></head><body><main>
<section class='card'><h1>Acrobot Swing-up Hybrid Control</h1><p>RLを使わず、energy-shaping seed → CasADi/IPOPT direct multiple shooting → upright LQR の順で真下からSwing-upと安定化を行います。</p><p class='ok'>Capture: {metrics['capture']} / Final stable: {metrics['final_stable']}</p><p><a href='{html.escape(run_url)}'>GitHub Actions run</a></p></section>
<section class='card'><h2>Result</h2><table><tbody>
<tr><td>Solver</td><td>{html.escape(str(summary['solver_status']))}</td></tr>
<tr><td>Trajectory → LQR switch</td><td>{summary['switch_time_s']:.2f} s</td></tr>
<tr><td>Capture time</td><td>{capture_time:.2f} s</td></tr>
<tr><td>Stable dwell</td><td>{metrics['stable_ratio']*100:.1f}%</td></tr>
<tr><td>Final 2 s stable</td><td>{metrics['final_2s_stable_ratio']*100:.1f}%</td></tr>
<tr><td>Final tip height</td><td>{metrics['final_tip_height_m']:.6f} m</td></tr>
<tr><td>Final theta1 error</td><td>{metrics['final_theta1_error_deg']:.6f} deg</td></tr>
<tr><td>Final theta2 error</td><td>{metrics['final_theta2_error_deg']:.6f} deg</td></tr>
<tr><td>RMS torque</td><td>{metrics['rms_torque_nm']:.3f} N m</td></tr>
</tbody></table></section>
<section class='card'><h2>Simulation</h2><video controls muted loop playsinline src='videos/hybrid.mp4'></video></section>
<section class='grid'><div class='card'><h2>Tip height</h2><img src='plots/height.png'></div><div class='card'><h2>Torque</h2><img src='plots/torque.png'></div></section>
<section class='card'><h2>States</h2><img src='plots/states.png'></section>
<section class='card'><h2>Interpretation</h2><p>最初の20秒は最適化されたbounded torque trajectoryを実行し、真上近傍で20秒のLQR保持へ切り替えます。ここをモデルベースの基準性能として、次段階でTVLQR/MPC・外乱耐性・RL Swing-upとのハイブリッドを追加します。</p></section>
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
