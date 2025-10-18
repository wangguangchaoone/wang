function SNRI = calculate_SNRI(input_signal, output_signal, target_freqs, fs)
% calculate_SNRI - Calculate Signal-to-Noise Ratio Improvement
%
% Implements Equation (9): SNRI = SNR_out / SNR_in
%
% Inputs:
%   input_signal - Input signal to the system
%   output_signal - Output signal from the system
%   target_freqs - Target signal frequencies (Hz)
%   fs - Sampling frequency (Hz)
%
% Outputs:
%   SNRI - Signal-to-noise ratio improvement

SNR_in = calculate_SNR(input_signal, target_freqs, fs);
SNR_out = calculate_SNR(output_signal, target_freqs, fs);

% SNRI (Equation 9)
SNRI = SNR_out / SNR_in;

end
