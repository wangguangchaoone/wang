function y = benchmark_ackley(x)
% Ackley function (min at 0) typical domain [-32.768, 32.768]
% x: 1xD vector
x = x(:)';
D = numel(x);
term1 = -20 * exp(-0.2 * sqrt(sum(x.^2) / D));
term2 = -exp(sum(cos(2*pi*x)) / D);
y = term1 + term2 + 20 + exp(1);
end
