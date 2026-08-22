# acrobot-swingup-hybrid-control

Continuous-torque Acrobot swing-up and stabilization with model-based / hybrid control.

This repository complements `rl-acrobot-swingup-benchmark`: instead of asking one learned policy to acquire swing-up and balance simultaneously, it separates trajectory generation, trajectory feedback, online model selection, nonlinear replanning, capture, and upright stabilization.

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

The current reference architecture is:

1. deterministic energy-shaping trajectory used only as an optimization seed,
2. CasADi/IPOPT direct multiple shooting over a **21 s** swing-up horizon,
3. nominal trajectory limited to **±0.98 N m** although the physical actuator can use ±1.00 N m,
4. finite-horizon discrete **TVLQR** feedback along all 1050 trajectory samples,
5. a **9-model uncertainty bank**: nominal plus the eight tested parameter-mismatch endpoints,
6. model selection from the first **0.5 s** of observed one-step state transitions,
7. constrained nonlinear replan for the selected model, warm-started from the successful nominal trajectory,
8. selected-model TVLQR during swing-up,
9. selected-model five-state upright LQR after the 21 s handoff,
10. 19 s additional upright hold.

Every non-nominal library trajectory is re-optimized under its candidate dynamics with a **±0.99 N m** trajectory limit and explicit terminal constraints: angle error within ±0.5° and angular velocity within ±0.05 rad/s. The physical plant remains limited to ±1.00 N m.

The online selector does not receive the scenario label. It compares candidate one-step predictions against the observed `(x, u, x_next)` transitions and selects the model with the smallest accumulated prediction error.

## Reproduced nominal result

PR #3 GitHub Actions solves the nominal trajectory, solves all eight constrained candidate replans, computes their TVLQR/LQR feedback, identifies models online, and runs the full robustness sweep from scratch.

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

At the nominal optimized trajectory terminal, before the local upright LQR takes over:

- theta1 error: `-0.0095 deg`,
- theta2 error: `-0.0046 deg`,
- dtheta1: `0.00050 rad/s`,
- dtheta2: `0.00033 rad/s`.

## Closed-loop robustness progression

| Perturbation group | Open-loop Final stable | Fixed TVLQR Final stable | Adaptive Final stable | Adaptive Capture |
|---|---:|---:|---:|---:|
| initial angle ±2° and initial velocity offsets | **0/8** | **8/8** | **8/8** | **8/8** |
| model mismatch | **0/8** | **3/8** | **8/8** | **8/8** |
| initial angle ±5° stress | **0/4** | **2/4** | **2/4** | **3/4** |

The eight model-mismatch cases are:

- mass / inertia ±2%,
- link length / COM ±1%,
- motor time constant ±10%,
- joint damping ±10%.

With fixed nominal TVLQR only **3/8** reached Final stable. With the multiple-model adaptive replanning layer, **all 8/8 reach Capture and Final stable**. The first-0.5-s model selector also identified the exact candidate **8/8** times in this noiseless endpoint test.

The original initial-state robustness gate remains unchanged: theta1/theta2 ±2°, dtheta1 ±0.05 rad/s, and dtheta2 ±0.08 rad/s must all return to Final stable. The new CI gate additionally requires at least 6/8 model-mismatch Final stable and 7/8 Capture; the reproduced PR #3 result exceeds that at 8/8 for both.

This result shows a clean hierarchy of the problem:

- open-loop optimization proves reachability,
- TVLQR closes ordinary local state error,
- online model selection + constrained nonlinear replanning closes the tested endpoint parameter mismatch,
- larger state deviations, between-bank parameters and noisy identification remain the next robustness targets.

## Outputs

`python -m acrobot_hybrid.run --output-dir results` creates:

- `summary.json` / `summary.md`,
- `trajectory.csv`,
- `nominal_trajectory.npz`,
- `tvlqr_gains.npy`,
- `robustness.csv` / `robustness.json`,
- `adaptive-library.json`,
- `plots/states.png`, `plots/height.png`, `plots/torque.png`,
- `videos/hybrid.mp4`.

CI runs the real nonlinear optimizers and the full robustness sweep. It fails unless nominal Capture + Final stable succeed, the original 8/8 initial-state TVLQR gate remains satisfied, and adaptive replanning materially improves the model-mismatch result.

## Next steps

The exact uncertainty-bank endpoints now reach 8/8, so the next tests must avoid simply matching a stored candidate exactly:

1. test **hold-out parameter values between the bank points** (for example mass +1.1%, tau +6%),
2. add measurement noise, bias and imperfect state estimation to the model selector,
3. replace one-shot model selection with continuously updated model probabilities / interpolation,
4. add true receding-horizon nonlinear MPC when the selected trajectory residual grows beyond the TVLQR funnel,
5. replace time-only swing-up → upright-LQR handoff with a hysteretic capture supervisor,
6. compare this model-based architecture with `RL swing-up + LQR/MPC balance` under identical physics and evaluation metrics.

## License

MIT
