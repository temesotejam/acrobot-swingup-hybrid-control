# acrobot-swingup-hybrid-control

Continuous-torque Acrobot swing-up and stabilization with model-based / hybrid control.

This repository complements `rl-acrobot-swingup-benchmark`: instead of asking one learned policy to acquire swing-up and balance simultaneously, it separates the problem into a swing-up phase, a capture/switching phase, and an upright stabilizer.

## First target

Use the same nominal plant as the RL benchmark (two 1 m / 1 kg links, elbow actuation only, ±1 N·m, 50 Hz, motor lag and joint damping) and demonstrate:

1. swing-up from the downward equilibrium,
2. entry into a capture region near the upright equilibrium,
3. switching to a local LQR stabilizer,
4. stable upright hold without terminating the episode,
5. reproducible simulation, plots, video and GitHub Pages output.

## Planned controllers

- deterministic energy-shaping baseline,
- local LQR stabilizer,
- hybrid supervisor with hysteresis,
- direct-collocation / trajectory optimization baseline,
- trajectory tracking with TVLQR,
- later: RL swing-up + model-based balance comparison.

## Status

Repository initialized. The first implementation will establish the plant model, hybrid baseline, automated tests and visualization pipeline.

## License

MIT
