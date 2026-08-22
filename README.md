# acrobot-swingup-hybrid-control

Continuous-torque Acrobot swing-up and stabilization with model-based / hybrid control.

This repository complements `rl-acrobot-swingup-benchmark`: instead of asking one learned policy to acquire swing-up and balance simultaneously, it separates trajectory generation, trajectory feedback, capture, and upright stabilization.

## Baseline plant

The nominal plant matches the RL benchmark:

- two serial 1.0 m / 1.0 kg links,
- shoulder unactuated, elbow actuated,
- physical continuous torque limit ±1.0 N m,
- 50 Hz integration (`dt=0.02 s`),
- first-order motor lag `tau=0.05 s`,
- joint viscous damping `0.02 N m/(rad/s)`,
- five-state model `[theta1, theta2, dtheta1, dtheta2, actual_torque]`.

## Current controller

The current reference controller is:

1. deterministic energy-shaping trajectory used only as an optimization seed,
2. CasADi/IPOPT direct multiple shooting over a **21 s** swing-up horizon,
3. nominal trajectory limited to **±0.98 N m** although the physical actuator can use ±1.00 N m,
4. numerical linearization of the exact RK4 step map along all 1050 swing-up samples,
5. finite-horizon discrete **TVLQR** feedback along the trajectory,
6. switch at 21 s to a local five-state upright LQR,
7. 19 s additional upright hold.

The 0.02 N m/side difference between nominal and physical torque limits deliberately leaves feedback headroom. Both TVLQR and the final LQR include the actuator torque state, so the 0.05 s motor lag is part of the feedback design.

## Reproduced result

PR #2 GitHub Actions solves the trajectory and computes all TVLQR gains from scratch before running the 40 s simulation and robustness sweep.

| Nominal metric | Result |
|---|---:|
| IPOPT status | `Solve_Succeeded` |
| Capture | **true** |
| Capture time | **19.12 s** |
| Final stable | **true** |
| Stable dwell over 40 s | **53.4%** |
| Final 2 s stable | **100.0%** |
| Final tip height | **2.000000 m** |
| RMS commanded torque | **0.685 N m** |

At the optimized trajectory terminal, before the local upright LQR takes over:

- theta1 error: `-0.0095 deg`,
- theta2 error: `-0.0046 deg`,
- dtheta1: `0.00050 rad/s`,
- dtheta2: `0.00033 rad/s`.

### Closed-loop robustness

The same nominal trajectory and the same nominal TVLQR gains are reused for every perturbation. The controller is **not** redesigned for the perturbed plant.

| Perturbation group | Open-loop Final stable | TVLQR Final stable | TVLQR Capture |
|---|---:|---:|---:|
| initial angle ±2° and initial velocity offsets | **0/8** | **8/8** | **8/8** |
| model mismatch | **0/8** | **3/8** | **6/8** |
| initial angle ±5° stress | **0/4** | **2/4** | **3/4** |

The initial-state group contains theta1/theta2 ±2°, dtheta1 ±0.05 rad/s, and dtheta2 ±0.08 rad/s. **All 8 must reach Final stable in CI.**

This is an important separation of problems: open-loop reachability was already established by PR #1; PR #2 shows that TVLQR creates a useful local closed-loop funnel around that trajectory. The remaining weakness is now mainly **model mismatch**, rather than ordinary small initial-state error.

## Outputs

`python -m acrobot_hybrid.run --output-dir results` creates:

- `summary.json` / `summary.md`,
- `trajectory.csv`,
- `nominal_trajectory.npz`,
- `tvlqr_gains.npy`,
- `robustness.csv` / `robustness.json`,
- `plots/states.png`, `plots/height.png`, `plots/torque.png`,
- `videos/hybrid.mp4`.

CI runs the real optimizer and the full robustness sweep. It fails unless nominal Capture + Final stable succeed and all 8 initial-state perturbations return to Final stable with TVLQR.

## Next steps

TVLQR has largely solved the small state-error problem, so the next priorities shift toward plant uncertainty and online adaptation:

1. add nonlinear MPC / short-horizon replanning around the nominal trajectory,
2. improve the current **3/8 Final stable** model-mismatch result without redesigning for the perturbed model,
3. replace time-only TVLQR → upright-LQR switching with a hysteretic capture supervisor,
4. add sensor noise / bias and state-estimation errors to robustness sweeps,
5. compare model-based swing-up with `RL swing-up + LQR/MPC balance` under identical physics and evaluation metrics.

## License

MIT
