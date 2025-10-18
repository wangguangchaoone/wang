%% Debug Test Script
% Quick test to verify the fixes

clear; clc; close all;

fprintf('========================================\n');
fprintf('Debug Test - TDCBSR Framework\n');
fprintf('========================================\n\n');

%% Test 1: Simple RK4 with known parameters
fprintf('Test 1: RK4 Solver with fixed parameters\n');
fprintf('------------------------------------------\n');

fs = 4.098;
N = 500;
t = (0:N-1)/fs;

% Simple test signal
f_test = 0.05;
A_test = 0.1;
signal = A_test * cos(2*pi*f_test*t);

% Add moderate noise
rng(42);
noise = randn(1, N) * 0.3;
input_signal = signal + noise;

% Fixed parameters (typical bistable system)
a = 1.0;
b = 1.0;
c = 1.0;
d = 1.0;
r = 0.5;
h = 1/fs;

fprintf('Parameters: a=%.2f, b=%.2f, c=%.2f, d=%.2f, r=%.2f\n', a, b, c, d, r);
fprintf('Signal: f=%.3fHz, A=%.2f, Noise std=%.2f\n', f_test, A_test, 0.3);

try
    [y, x] = TDCBSR_RK4(input_signal, a, b, c, d, r, h);
    fprintf('✓ RK4 completed successfully\n');
    fprintf('  Output y range: [%.4f, %.4f]\n', min(y), max(y));
    fprintf('  Output x range: [%.4f, %.4f]\n', min(x), max(x));
    fprintf('  NaN count: %d\n', sum(isnan(y)));
    fprintf('  Inf count: %d\n', sum(isinf(y)));
    
    if any(isnan(y)) || any(isinf(y))
        fprintf('✗ ERROR: Output contains NaN or Inf!\n\n');
        return;
    end
    
catch ME
    fprintf('✗ ERROR: %s\n\n', ME.message);
    return;
end

%% Test 2: SNR Calculation
fprintf('\nTest 2: SNR Calculation\n');
fprintf('------------------------------------------\n');

try
    SNR_in = calculate_SNR(input_signal, [f_test], fs);
    SNR_out = calculate_SNR(y, [f_test], fs);
    
    fprintf('Input SNR:  %.2f dB\n', SNR_in);
    fprintf('Output SNR: %.2f dB\n', SNR_out);
    
    if SNR_in > -1000 && SNR_out > -1000
        fprintf('✓ SNR calculation successful\n');
        SNRI = SNR_out / SNR_in;
        fprintf('  SNRI = %.4f\n', SNRI);
    else
        fprintf('✗ ERROR: SNR calculation returned penalty value\n\n');
        return;
    end
    
catch ME
    fprintf('✗ ERROR: %s\n\n', ME.message);
    return;
end

%% Test 3: Visualize Results
fprintf('\nTest 3: Visualization\n');
fprintf('------------------------------------------\n');

figure('Position', [100, 100, 1200, 400]);

% Time domain
subplot(1, 3, 1);
plot(t, input_signal, 'k', 'LineWidth', 0.5);
hold on;
plot(t, signal, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Input Signal');
legend('Noisy', 'Clean');
grid on;

subplot(1, 3, 2);
plot(t, y, 'b', 'LineWidth', 0.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('TDCBSR Output (y)');
grid on;

% Frequency domain comparison
subplot(1, 3, 3);
N_fft = length(input_signal);
freq = (0:N_fft-1) * (fs/N_fft);
freq_half = freq(1:floor(N_fft/2)+1);

Y_in = fft(input_signal);
P_in = abs(Y_in/N_fft);
P_in_half = P_in(1:floor(N_fft/2)+1);

Y_out = fft(y);
P_out = abs(Y_out/N_fft);
P_out_half = P_out(1:floor(N_fft/2)+1);

plot(freq_half, P_in_half, 'k-', 'LineWidth', 1);
hold on;
plot(freq_half, P_out_half, 'b-', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Spectrum');
legend('Input', 'Output');
xlim([0, 0.15]);
grid on;

fprintf('✓ Visualization complete\n');

%% Test 4: Quick BSO Test
fprintf('\nTest 4: Quick BSO Optimization (5 iterations)\n');
fprintf('------------------------------------------\n');

param_ranges.a = [0.5, 2.5];
param_ranges.b = [0.5, 2.5];
param_ranges.c = [0.5, 2.5];
param_ranges.d = [0.5, 2.5];
param_ranges.r = [0.1, 1.5];

try
    [best_params, best_SNR] = BSO_optimization(input_signal, [f_test], fs, h, ...
                                               param_ranges, 5, 10, 3);
    
    fprintf('Best SNR: %.2f dB\n', best_SNR);
    
    if best_SNR > -1000
        fprintf('✓ BSO optimization successful\n');
        fprintf('  Optimized parameters:\n');
        fprintf('    a=%.4f, b=%.4f, c=%.4f, d=%.4f, r=%.4f\n', ...
                best_params(1), best_params(2), best_params(3), ...
                best_params(4), best_params(5));
    else
        fprintf('✗ WARNING: BSO returned penalty value\n');
        fprintf('  This may indicate numerical instability issues\n');
    end
    
catch ME
    fprintf('✗ ERROR: %s\n\n', ME.message);
    return;
end

fprintf('\n========================================\n');
fprintf('Debug Test Complete\n');
fprintf('========================================\n');
fprintf('\nIf all tests passed, the framework is ready!\n');
fprintf('If BSO still returns -1000, consider:\n');
fprintf('  1. Adjusting parameter ranges\n');
fprintf('  2. Increasing time step (reducing temporal resolution)\n');
fprintf('  3. Reducing noise intensity\n');
fprintf('  4. Checking signal scaling\n');
