function [y, x] = TDCBSR_RK4(input_signal, a, b, c, d, r, h)
% TDCBSR_RK4 - Two-Dimensional Coupled Bistable Stochastic Resonance
% using 4th-order Runge-Kutta method
%
% Implements Equation (2) and (4) from the paper
% dU(y)/dt = -U'(y) + r(y-x) + s(t) + η(t)
% dV(x)/dt = -V'(x) + r(x-y)
%
% where U(y) = -(a/2)*y^2 + (b/4)*y^4
%       V(x) = -(c/2)*x^2 + (d/4)*x^4
%
% Inputs:
%   input_signal - Input signal with noise s(t) + η(t)
%   a, b, c, d - System parameters
%   r - Coupling coefficient
%   h - Time step
%
% Outputs:
%   y - Output from controlled system
%   x - Output from controlling system

N = length(input_signal);
y = zeros(1, N);
x = zeros(1, N);

% Initial conditions
y(1) = 0;
x(1) = 0;

% Runge-Kutta 4th order integration
for n = 1:(N-1)
    u = input_signal(n);
    
    % k1, g1
    k1 = h * (a*y(n) - b*y(n)^3 + r*(y(n) - x(n)) + u);
    g1 = h * (c*x(n) - d*x(n)^3 + r*(x(n) - y(n)));
    
    % k2, g2
    k2 = h * (a*(y(n) + k1/2) - b*(y(n) + k1/2)^3 + ...
              r*((y(n) + g1/2) - (x(n) + k1/2)) + u);
    g2 = h * (c*(x(n) + g1/2) - d*(x(n) + g1/2)^3 + ...
              r*((x(n) + k1/2) - (y(n) + g1/2)));
    
    % k3, g3
    k3 = h * (a*(y(n) + k2/2) - b*(y(n) + k2/2)^3 + ...
              r*((y(n) + g2/2) - (x(n) + k2/2)) + u);
    g3 = h * (c*(x(n) + g2/2) - d*(x(n) + g2/2)^3 + ...
              r*((x(n) + k2/2) - (y(n) + g2/2)));
    
    % k4, g4
    k4 = h * (a*(y(n) + k3) - b*(y(n) + k3)^3 + ...
              r*((y(n) + g3) - (x(n) + k3)) + u);
    g4 = h * (c*(x(n) + g3) - d*(x(n) + g3)^3 + ...
              r*((x(n) + k3) - (y(n) + g3)));
    
    % Update
    x(n+1) = x(n) + (1/6) * (k1 + 2*k2 + 2*k3 + k4);
    y(n+1) = y(n) + (1/6) * (g1 + 2*g2 + 2*g3 + g4);
end

end
