%% High-Frequency Signal Detection using TDCBSR-BSO
% This script implements the detection of multiple high-frequency periodic signals
% as described in the paper and shown in Fig. 16

clear; clc; close all;

%% Signal parameters (from paper)
% Multiple high-frequency weak periodic signals
f1 = 100;     % Hz
f2 = 300;     % Hz
f3 = 500;     % Hz
A1 = 0.05;    % Signal amplitude
A2 = 0.05;
A3 = 0.05;

fs = 20480;   % Sampling frequency (Hz) - from paper fs=20480Hz
N = 4096;     % Number of sampling points
t = (0:N-1) / fs;  % Time vector

% Generate clean signal
s = A1*cos(2*pi*f1*t) + A2*cos(2*pi*f2*t) + A3*cos(2*pi*f3*t);

%% Noise parameters (α-stable noise)
% α-stable distribution parameters
alpha = 1.2;  % Stability parameter
beta = 0;     % Skewness
sigma = 1;    % Scale
mu = 0.5;     % Location

% Generate α-stable noise
rng(123);  % For reproducibility
noise = generate_alpha_stable_noise(N, alpha, beta, sigma, mu);

% Input signal with noise
D = 0.5;  % Noise intensity
G = 10000;  % Parameter compensation (from paper)
input_signal = s + D * noise;

%% BSO optimization parameters
% For high-frequency signals, use scaled parameter ranges
param_ranges.a = [0.5, 3.0];    % Range for parameter a
param_ranges.b = [0.5, 3.0];    % Range for parameter b
param_ranges.c = [0.5, 3.0];    % Range for parameter c
param_ranges.d = [0.5, 3.0];    % Range for parameter d
param_ranges.r = [0.1, 2.0];    % Range for coupling coefficient r

max_iter = 100;   % Maximum iterations
pop_size = 30;    % Population size
n_clusters = 5;   % Number of clusters

% Time step for RK4
h = 1/fs;

% Target frequencies for SNR calculation
target_freqs = [f1, f2, f3];

fprintf('Starting BSO optimization for high-frequency signal detection...\n');
fprintf('Target frequencies: %d Hz, %d Hz, %d Hz\n', f1, f2, f3);

%% Run BSO optimization
[best_params, best_SNR] = BSO_optimization(input_signal, target_freqs, fs, h, ...
                                           param_ranges, max_iter, pop_size, n_clusters);

%% Apply TDCBSR with optimized parameters
a_opt = best_params(1);
b_opt = best_params(2);
c_opt = best_params(3);
d_opt = best_params(4);
r_opt = best_params(5);

[output_signal, ~] = TDCBSR_RK4(input_signal, a_opt, b_opt, c_opt, d_opt, r_opt, h);

%% Calculate SNRI
SNRI = calculate_SNRI(input_signal, output_signal, target_freqs, fs);
fprintf('SNRI = %.2f\n', SNRI);

%% Plotting results (Fig. 16)
figure('Position', [100, 100, 1200, 800]);

% Input signal time-domain
subplot(2, 2, 1);
plot(t, input_signal, 'k', 'LineWidth', 0.5);
xlabel('t/s', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Input signal time-domain diagram', 'FontSize', 12);
xlim([0, 0.2]);
grid on;

% Input signal frequency-domain
subplot(2, 2, 2);
N_fft = length(input_signal);
Y_in = fft(input_signal);
P_in = abs(Y_in/N_fft);
freq = (0:N_fft-1) * (fs/N_fft);
freq_half = freq(1:floor(N_fft/2)+1);
P_in_half = P_in(1:floor(N_fft/2)+1);
plot(freq_half, P_in_half, 'k', 'LineWidth', 1);
xlabel('f/Hz', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Input signal frequency-domain diagram', 'FontSize', 12);
xlim([0, 2000]);
grid on;

% Output signal time-domain
subplot(2, 2, 3);
plot(t, output_signal, 'b', 'LineWidth', 0.5);
xlabel('t/s', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Output signal time-domain diagram', 'FontSize', 12);
xlim([0, 0.2]);
grid on;

% Output signal frequency-domain
subplot(2, 2, 4);
Y_out = fft(output_signal);
P_out = abs(Y_out/N_fft);
P_out_half = P_out(1:floor(N_fft/2)+1);
plot(freq_half, P_out_half, 'b', 'LineWidth', 1);
hold on;
% Mark detected frequencies
[~, idx1] = min(abs(freq_half - f1));
[~, idx2] = min(abs(freq_half - f2));
[~, idx3] = min(abs(freq_half - f3));
plot(freq_half(idx1), P_out_half(idx1), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
plot(freq_half(idx2), P_out_half(idx2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
plot(freq_half(idx3), P_out_half(idx3), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
text(freq_half(idx1)+50, P_out_half(idx1), sprintf('%dHz', round(freq_half(idx1))), 'FontSize', 10);
text(freq_half(idx2)+50, P_out_half(idx2), sprintf('%dHz', round(freq_half(idx2))), 'FontSize', 10);
text(freq_half(idx3)+50, P_out_half(idx3), sprintf('%dHz', round(freq_half(idx3))), 'FontSize', 10);
xlabel('f/Hz', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Output signal frequency-domain diagram', 'FontSize', 12);
xlim([0, 2000]);
grid on;

sgtitle('Detection results of adaptive TDCBSR for multiple high-frequency periodic signals under α-stable noise', ...
        'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'Fig16_high_frequency_detection.png');
fprintf('Results saved to Fig16_high_frequency_detection.png\n');
