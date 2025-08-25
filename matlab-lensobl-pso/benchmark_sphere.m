function y = benchmark_sphere(x)
% Sphere function (sum of squares), min at 0
x = x(:)';
y = sum(x.^2);
end
