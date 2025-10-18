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
    
    % k1 for y, g1 for x
    % dy/dt = a*y - b*y^3 + r*(y-x) + u
    % dx/dt = c*x - d*x^3 + r*(x-y)
    k1 = h * (a*y(n) - b*y(n)^3 + r*(y(n) - x(n)) + u);
    g1 = h * (c*x(n) - d*x(n)^3 + r*(x(n) - y(n)));
    
    % k2 for y, g2 for x
    y_temp = y(n) + k1/2;
    x_temp = x(n) + g1/2;
    k2 = h * (a*y_temp - b*y_temp^3 + r*(y_temp - x_temp) + u);
    g2 = h * (c*x_temp - d*x_temp^3 + r*(x_temp - y_temp));
    
    % k3 for y, g3 for x
    y_temp = y(n) + k2/2;
    x_temp = x(n) + g2/2;
    k3 = h * (a*y_temp - b*y_temp^3 + r*(y_temp - x_temp) + u);
    g3 = h * (c*x_temp - d*x_temp^3 + r*(x_temp - y_temp));
    
    % k4 for y, g4 for x
    y_temp = y(n) + k3;
    x_temp = x(n) + g3;
    k4 = h * (a*y_temp - b*y_temp^3 + r*(y_temp - x_temp) + u);
    g4 = h * (c*x_temp - d*x_temp^3 + r*(x_temp - y_temp));
    
    % Update: k updates y, g updates x
    y(n+1) = y(n) + (1/6) * (k1 + 2*k2 + 2*k3 + k4);
    x(n+1) = x(n) + (1/6) * (g1 + 2*g2 + 2*g3 + g4);
    
    % Check for numerical instability
    if isnan(y(n+1)) || isinf(y(n+1)) || isnan(x(n+1)) || isinf(x(n+1))
        warning('TDCBSR_RK4: Numerical instability detected at step %d', n);
        % Clip to prevent further instability
        if isnan(y(n+1)) || isinf(y(n+1))
            y(n+1) = y(n);
        end
        if isnan(x(n+1)) || isinf(x(n+1))
            x(n+1) = x(n);
        end
    end
    
    % Clip extremely large values to prevent overflow
    max_val = 1e6;
    y(n+1) = max(-max_val, min(max_val, y(n+1)));
    x(n+1) = max(-max_val, min(max_val, x(n+1)));
end

end
