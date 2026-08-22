# acrobot-swingup-hybrid-control

Continuous-torque Acrobot swing-up and stabilization with model-based / hybrid control.

This repository complements `rl-acrobot-swingup-benchmark`: instead of asking one learned policy to acquire swing-up and balance simultaneously, it separates the problem into a swing-up phase, a capture/switching phase, and an upright stabilizer.

## Baseline plant

The nominal plant matches the RL benchmark:

- two serial 1.0 m / 1.0 kg links,
- shoulder unactuated, elbow actuated,
- continuous torque limited to ±1.0 N m,
- 50 Hz integration (`dt=0.02 s`),
- first-order motor lag `tau=0.05 s`,
- joint viscous damping `0.02 N m/(rad/s)`,
- five-state model `[theta1, theta2, dtheta1, dtheta2, actual_torque]`.

## Current controller

The first working reference controller is:

1. deterministic energy-shaping trajectory used only as an optimization seed,
2. CasADi/IPOPT direct multiple shooting over a 20 s swing-up horizon,
3. bounded torque trajectory with the same ±1 N m limit,
4. switch at 20 s to a local five-state LQR around the upright equilibrium,
5. 20 s additional upright hold.

The LQR design includes the actuator torque state, so the 0.05 s motor lag is not ignored during stabilization.

## Reproduced result

PR #1 GitHub Actions solved the trajectory from scratch and then ran the full 40 s simulation.

| Metric | Result |
|---|---:|
| IPOPT status | `Solve_Succeeded` |
| Capture | **true** |
| Capture time | **18.92 s** |
| Final stable | **true** |
| Stable dwell over 40 s | **53.9%** |
| Final 2 s stable | **100.0%** |
| Final tip height | **2.000000 m** |
| Final theta1 error | **0.000000 deg** |
| Final theta2 error | **~0.000000 deg** |
| RMS commanded torque | **0.692 N m** |

At the trajectory-optimization terminal, before LQR takes over:

- theta1 error: `-0.3020 deg`,
- theta2 error: `-0.1721 deg`,
- dtheta1: `0.01674 rad/s`,
- dtheta2: `0.01203 rad/s`.

This establishes that the nominal ±1 N m plant is capable of full swing-up and long upright stabilization without RL.

## Outputs

`python -m acrobot_hybrid.run --output-dir results` creates:

- `summary.json` / `summary.md`,
- `trajectory.csv`,
- `nominal_trajectory.npz`,
- `plots/states.png`, `plots/height.png`, `plots/torque.png`,
- `videos/hybrid.mp4`.

CI runs the real optimizer, not a mocked or cached answer, and fails unless both Capture and Final stable succeed.

## Next steps

The nominal open-loop swing-up trajectory is now a reference, not the final architecture. Next priorities are:

1. add TVLQR / trajectory feedback during the swing-up phase,
2. quantify robustness to initial-angle, velocity, parameter and sensor perturbations,
3. replace time-only switching with a hysteretic capture supervisor,
4. add nonlinear MPC as a second model-based controller,
5. compare model-based swing-up with `RL swing-up + LQR/MPC balance` under identical physics and evaluation metrics.

## License

MIT
