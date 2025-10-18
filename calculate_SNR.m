function SNR = calculate_SNR(output_signal, target_freqs, fs)
% calculate_SNR - Calculate Signal-to-Noise Ratio
%
% Implements Equation (8): SNR = 10*log10(P_signal/P_noise)
%
% Inputs:
%   output_signal - Output signal from the system
%   target_freqs - Target signal frequencies (Hz)
%   fs - Sampling frequency (Hz)
%
% Outputs:
%   SNR - Signal-to-noise ratio in dB

N = length(output_signal);
Y = fft(output_signal);
P = abs(Y/N).^2;  % Power spectrum

% Frequency resolution
freq = (0:N-1) * (fs/N);
freq = freq(1:floor(N/2)+1);
P = P(1:floor(N/2)+1);

% Find power at signal frequencies
P_signal = 0;
for i = 1:length(target_freqs)
    [~, idx] = min(abs(freq - target_freqs(i)));
    % Sum power in a small window around the target frequency
    window = max(1, idx-2):min(length(P), idx+2);
    P_signal = P_signal + sum(P(window));
end

% Calculate noise power (total power minus signal power)
P_total = sum(P);
P_noise = P_total - P_signal;

% Avoid division by zero
if P_noise == 0
    P_noise = eps;
end

% SNR in dB (Equation 8)
SNR = 10 * log10(P_signal / P_noise);

end
