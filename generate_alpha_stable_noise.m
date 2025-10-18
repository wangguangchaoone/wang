function noise = generate_alpha_stable_noise(N, alpha, beta, sigma, mu)
% generate_alpha_stable_noise - Generate α-stable distributed noise
%
% This function generates noise following an α-stable distribution
% using the Chambers-Mallows-Stuck method
%
% Inputs:
%   N - Number of samples
%   alpha - Stability parameter (0 < alpha <= 2)
%   beta - Skewness parameter (-1 <= beta <= 1)
%   sigma - Scale parameter (sigma > 0)
%   mu - Location parameter
%
% Output:
%   noise - α-stable distributed noise vector

% Generate uniform random variable
U = (rand(1, N) - 0.5) * pi;

% Generate exponential random variable
W = -log(rand(1, N));

% Special case: alpha = 2 (Gaussian)
if alpha == 2
    noise = sqrt(2) * sigma * randn(1, N) + mu;
    return;
end

% Special case: alpha = 1 (Cauchy)
if alpha == 1
    if beta == 0
        noise = sigma * tan(U) + mu;
    else
        noise = sigma * tan(U) + (2/pi) * beta * sigma * log(sigma * W) + mu;
    end
    return;
end

% General case: Chambers-Mallows-Stuck method
B = atan(beta * tan(pi * alpha / 2)) / alpha;
S = (1 + beta^2 * tan(pi * alpha / 2)^2)^(1/(2*alpha));

if abs(alpha - 1) > 1e-10
    noise = S * sin(alpha * (U + B)) ./ (cos(U).^(1/alpha)) .* ...
            (cos(U - alpha * (U + B)) ./ W).^((1-alpha)/alpha);
    noise = sigma * noise + mu;
else
    % For alpha close to 1
    noise = sigma * tan(U) + (2/pi) * beta * sigma * log(sigma * W) + mu;
end

end
