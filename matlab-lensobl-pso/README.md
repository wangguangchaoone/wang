# Lens-OBL Based PSO with SR (MATLAB)

This project implements a Particle Swarm Optimization (PSO) algorithm enhanced with Lens Opposition-Based Learning (Lens-OBL) and a Selective Randomization (SR) mechanism, following the provided flowchart.

## Run

Open MATLAB, `cd` into the project folder and run:

```matlab
main
```

Choose the objective in `main.m` by setting `objective.name` to `'ackley'` or `'sphere'` and adjust parameters as needed.

## Files

- `main.m`: Configures problem and runs the algorithm
- `pso_lensobl.m`: Core PSO + Lens-OBL + SR loop
- `lens_opposition.m`: Lens-OBL operator
- `sr_operator.m`: Selective randomization around the global best
- `update_velocity_position.m`: Standard PSO update step with clamping
- `benchmark_ackley.m`, `benchmark_sphere.m`: Example objective functions
- `utils_plot.m`: Log-scale plot of best fitness over iterations

## Notes

- Bounds are per-dimension and set from the chosen benchmark.
- SR triggers when the swarm spreads too widely (sigma >> A), improvement is tiny (l << 1), or velocity amplitude is too large (a too large), and also after a configurable stagnation period.
- All parameters are defined in `main.m` under `params`.
