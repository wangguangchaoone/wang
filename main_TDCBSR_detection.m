%% Main TDCBSR Detection Framework
% Two-Dimensional Coupled Bistable Stochastic Resonance (TDCBSR)
% with Brain Storm Optimization (BSO) for weak signal detection
%
% This is the main entry point for the detection framework

clear; clc; close all;

fprintf('========================================\n');
fprintf('TDCBSR-BSO Detection Framework\n');
fprintf('========================================\n\n');

fprintf('This framework implements:\n');
fprintf('1. Two-Dimensional Coupled Bistable Stochastic Resonance (TDCBSR)\n');
fprintf('2. Brain Storm Optimization (BSO) for parameter optimization\n');
fprintf('3. Signal-to-Noise Ratio (SNR) evaluation\n\n');

fprintf('Available detection scripts:\n');
fprintf('1. detect_low_frequency_signals.m  - Low-frequency signal detection (Fig. 15)\n');
fprintf('2. detect_high_frequency_signals.m - High-frequency signal detection (Fig. 16)\n\n');

% Ask user which detection to run
fprintf('Select detection mode:\n');
fprintf('[1] Low-frequency signals (0.01Hz, 0.03Hz, 0.05Hz)\n');
fprintf('[2] High-frequency signals (100Hz, 300Hz, 500Hz)\n');
fprintf('[3] Both\n');

choice = input('Enter your choice (1-3): ');

switch choice
    case 1
        fprintf('\n--- Running Low-Frequency Signal Detection ---\n');
        run('detect_low_frequency_signals.m');
        
    case 2
        fprintf('\n--- Running High-Frequency Signal Detection ---\n');
        run('detect_high_frequency_signals.m');
        
    case 3
        fprintf('\n--- Running Low-Frequency Signal Detection ---\n');
        run('detect_low_frequency_signals.m');
        
        fprintf('\n\n--- Running High-Frequency Signal Detection ---\n');
        run('detect_high_frequency_signals.m');
        
    otherwise
        fprintf('Invalid choice. Running low-frequency detection by default.\n');
        run('detect_low_frequency_signals.m');
end

fprintf('\n========================================\n');
fprintf('Detection completed!\n');
fprintf('========================================\n');
