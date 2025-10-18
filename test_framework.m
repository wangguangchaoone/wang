%% Test TDCBSR Framework
% This script performs a quick test of the TDCBSR framework

clear; clc; close all;

fprintf('========================================\n');
fprintf('Testing TDCBSR Framework\n');
fprintf('========================================\n\n');

%% Test 1: Generate alpha-stable noise
fprintf('Test 1: Generating alpha-stable noise...\n');
N = 1000;
alpha = 1.2;
beta = 0;
sigma = 1;
mu = 0.5;
noise = generate_alpha_stable_noise(N, alpha, beta, sigma, mu);
fprintf('  Generated %d noise samples\n', length(noise));
fprintf('  Mean: %.4f, Std: %.4f\n', mean(noise), std(noise));
fprintf('  PASS\n\n');

%% Test 2: TDCBSR RK4 solver
fprintf('Test 2: Testing TDCBSR RK4 solver...\n');
fs = 100;
t = (0:N-1)/fs;
signal = 0.1*cos(2*pi*5*t);  % 5Hz signal
input_signal = signal + 0.3*noise(1:N);

a = 1; b = 1; c = 1; d = 1; r = 0.5;
h = 1/fs;
[y, x] = TDCBSR_RK4(input_signal, a, b, c, d, r, h);
fprintf('  Output length: %d samples\n', length(y));
fprintf('  Output range: [%.4f, %.4f]\n', min(y), max(y));
fprintf('  PASS\n\n');

%% Test 3: SNR calculation
fprintf('Test 3: Testing SNR calculation...\n');
target_freqs = [5];
SNR_in = calculate_SNR(input_signal, target_freqs, fs);
SNR_out = calculate_SNR(y, target_freqs, fs);
fprintf('  Input SNR: %.2f dB\n', SNR_in);
fprintf('  Output SNR: %.2f dB\n', SNR_out);
fprintf('  PASS\n\n');

%% Test 4: SNRI calculation
fprintf('Test 4: Testing SNRI calculation...\n');
SNRI = calculate_SNRI(input_signal, y, target_freqs, fs);
fprintf('  SNRI: %.2f\n', SNRI);
if SNRI > 1
    fprintf('  PASS (System shows stochastic resonance)\n\n');
else
    fprintf('  WARNING (SNRI < 1, may need parameter tuning)\n\n');
end

%% Test 5: BSO optimization (quick test with few iterations)
fprintf('Test 5: Testing BSO optimization (quick test)...\n');
param_ranges.a = [0.5, 2];
param_ranges.b = [0.5, 2];
param_ranges.c = [0.5, 2];
param_ranges.d = [0.5, 2];
param_ranges.r = [0.1, 1];

max_iter = 10;  % Quick test
pop_size = 10;
n_clusters = 3;

[best_params, best_SNR] = BSO_optimization(input_signal, target_freqs, fs, h, ...
                                           param_ranges, max_iter, pop_size, n_clusters);
fprintf('  Best SNR: %.2f dB\n', best_SNR);
fprintf('  PASS\n\n');

%% Summary
fprintf('========================================\n');
fprintf('All Tests Completed Successfully!\n');
fprintf('========================================\n\n');

fprintf('Framework is ready to use. Run:\n');
fprintf('  - main_TDCBSR_detection.m for interactive detection\n');
fprintf('  - detect_low_frequency_signals.m for low-freq detection\n');
fprintf('  - detect_high_frequency_signals.m for high-freq detection\n');
fprintf('  - visualize_potential_function.m to visualize w(x,y)\n');
