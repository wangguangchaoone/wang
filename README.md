# TDCBSR-BSO Detection Framework

This MATLAB implementation provides a complete framework for detecting weak signals in noisy environments using Two-Dimensional Coupled Bistable Stochastic Resonance (TDCBSR) combined with Brain Storm Optimization (BSO).

## Overview

The framework implements the TDCBSR system described in the research paper, which enhances weak signal detection performance by:
- Utilizing a two-dimensional coupled bistable stochastic resonance system
- Adaptive parameter optimization using BSO algorithm
- Multi-parameter collaborative optimization for improved SNR

## Features

- **TDCBSR System**: Two-dimensional coupled bistable model with flexible switching between monostable, bistable, and quad-stable states
- **BSO Optimization**: Adaptive parameter selection to maximize output SNR
- **Signal Detection**: Detects multiple periodic signals under α-stable noise
- **Performance Metrics**: SNR and SNRI calculation for system evaluation

## File Structure

```
TDCBSR_RK4.m                      - Runge-Kutta 4th order solver for TDCBSR system (Eq. 2, 4)
BSO_optimization.m                 - Brain Storm Optimization algorithm
calculate_SNR.m                    - Signal-to-Noise Ratio calculation (Eq. 8)
calculate_SNRI.m                   - SNR Improvement calculation (Eq. 9)
generate_alpha_stable_noise.m      - α-stable noise generator
detect_low_frequency_signals.m     - Low-frequency signal detection (Fig. 15)
detect_high_frequency_signals.m    - High-frequency signal detection (Fig. 16)
main_TDCBSR_detection.m           - Main entry point
```

## Theory

### TDCBSR System Equations

The system is governed by the two-dimensional Langevin equation:

```
dU(y)/dt = -U'(y) + r(y-x) + s(t) + η(t)
dV(x)/dt = -V'(x) + r(x-y)
```

where:
- `U(y) = -(a/2)*y² + (b/4)*y⁴` (controlled system)
- `V(x) = -(c/2)*x² + (d/4)*x⁴` (controlling system)
- `r` is the coupling coefficient
- `s(t)` is the target signal
- `η(t)` is α-stable noise

### Equivalent Potential Function

```
w(x,y) = (b/4)*y⁴ + (d/4)*x⁴ - (1/2)*(c+r)*x² - (1/2)*(r+a)*y² + rxy
```

### Signal-to-Noise Ratio (SNR)

```
SNR = 10*log₁₀(P_signal / P_noise)
```

### SNR Improvement (SNRI)

```
SNRI = SNR_out / SNR_in
```

## Usage

### Quick Start

Run the main script:
```matlab
main_TDCBSR_detection
```

Select detection mode:
1. Low-frequency signals (0.01Hz, 0.03Hz, 0.05Hz)
2. High-frequency signals (100Hz, 300Hz, 500Hz)
3. Both

### Individual Detection Scripts

**Low-Frequency Detection:**
```matlab
detect_low_frequency_signals
```
- Detects signals at 0.01Hz, 0.03Hz, 0.05Hz
- Sampling frequency: 4.098 Hz
- 4096 sampling points
- Reproduces Fig. 15 from the paper

**High-Frequency Detection:**
```matlab
detect_high_frequency_signals
```
- Detects signals at 100Hz, 300Hz, 500Hz
- Sampling frequency: 20480 Hz
- 4096 sampling points
- Reproduces Fig. 16 from the paper

### Custom Detection

You can create custom detection by calling the functions directly:

```matlab
% Setup signal and noise
fs = 1000;  % Sampling frequency
N = 4096;   % Number of points
t = (0:N-1)/fs;

% Generate signal with noise
signal = 0.05*cos(2*pi*10*t);  % 10Hz signal
noise = generate_alpha_stable_noise(N, 1.2, 0, 1, 0.5);
input_signal = signal + 0.5*noise;

% Define parameter ranges
param_ranges.a = [0.1, 5];
param_ranges.b = [0.1, 5];
param_ranges.c = [0.1, 5];
param_ranges.d = [0.1, 5];
param_ranges.r = [0.01, 2];

% Run BSO optimization
[best_params, best_SNR] = BSO_optimization(input_signal, [10], fs, 1/fs, ...
                                           param_ranges, 100, 30, 5);

% Apply TDCBSR
[output, ~] = TDCBSR_RK4(input_signal, best_params(1), best_params(2), ...
                         best_params(3), best_params(4), best_params(5), 1/fs);
```

## Parameters

### System Parameters
- `a, b`: Controlled system parameters (U(y))
- `c, d`: Controlling system parameters (V(x))
- `r`: Coupling coefficient between subsystems

### BSO Parameters
- `max_iter`: Maximum iterations (default: 100)
- `pop_size`: Population size (default: 30)
- `n_clusters`: Number of clusters for k-means (default: 5)

### Noise Parameters
- `alpha`: Stability parameter (0 < α ≤ 2)
- `beta`: Skewness parameter (-1 ≤ β ≤ 1)
- `sigma`: Scale parameter
- `mu`: Location parameter

## Output

The detection scripts generate:
1. **Time-domain plots**: Input and output signals
2. **Frequency-domain plots**: FFT spectra with detected frequencies marked
3. **Console output**: Optimized parameters, SNR, and SNRI values
4. **Saved figures**: PNG files (Fig15_low_frequency_detection.png, Fig16_high_frequency_detection.png)

## Algorithm Flow

1. **Population Initialization**: Generate random parameter sets
2. **K-means Clustering**: Divide population into clusters
3. **Fitness Calculation**: Calculate SNR for each individual
4. **Mutation and Update**: Generate new individuals through cluster operations
5. **Iteration**: Repeat until convergence or max iterations
6. **Output**: Best parameters and corresponding SNR

## Requirements

- MATLAB R2016b or later
- Statistics and Machine Learning Toolbox (for k-means clustering)

## Performance

The TDCBSR-BSO framework can:
- Detect extremely weak signals with very low SNR
- Handle multiple frequency components simultaneously
- Adapt to different noise conditions (α-stable distributions)
- Outperform traditional one-dimensional stochastic resonance methods

## References

Based on the research paper on Two-Dimensional Coupled Bistable Stochastic Resonance and BSO algorithm for weak fault signal detection.

## Notes

- The optimization process may take several minutes depending on signal length and iteration count
- For better results, adjust parameter ranges based on signal characteristics
- The α-stable noise generator uses the Chambers-Mallows-Stuck method
- Results may vary slightly due to random initialization in BSO

## License

Academic and research use.
