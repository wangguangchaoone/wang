%% Visualize TDCBSR Potential Function
% This script visualizes the equivalent potential function w(x,y)
% as shown in Fig. 2 of the paper

clear; clc; close all;

% System parameters (example values)
a = 1;
b = 1;
c = 1;
d = 1;
r = 0.5;

% Create mesh grid
x = linspace(-6, 6, 100);
y = linspace(-6, 6, 100);
[X, Y] = meshgrid(x, y);

% Calculate potential function w(x,y) - Equation (3)
% w(x,y) = (b/4)*y^4 + (d/4)*x^4 - (1/2)*(c+r)*x^2 - (1/2)*(r+a)*y^2 + r*x*y
W = (b/4)*Y.^4 + (d/4)*X.^4 - (1/2)*(c+r)*X.^2 - (1/2)*(r+a)*Y.^2 + r*X.*Y;

%% 3D visualization
figure('Position', [100, 100, 1400, 600]);

% (a) 3-dimensional diagram
subplot(1, 2, 1);
surf(X, Y, W, 'EdgeColor', 'none');
colormap('jet');
colorbar;
xlabel('x', 'FontSize', 12);
ylabel('y', 'FontSize', 12);
zlabel('w(x,y)', 'FontSize', 12);
title('(a) 3-dimensional diagram of w(x,y)', 'FontSize', 14);
view(-30, 30);
shading interp;
lighting gouraud;
light;

% (b) Vertical view (contour plot)
subplot(1, 2, 2);
contourf(X, Y, W, 50, 'LineColor', 'none');
colormap('jet');
colorbar;
hold on;

% Find and mark stable points (local minima)
% These are approximate locations - in practice, you'd use optimization
stable_points = [
    -3, -3;   % S1 (PW1)
     3, -3;   % S2 (PW2)
    -3,  3;   % S3 (PW3)
     3,  3    % S4 (PW4)
];

% Mark stable points
plot(stable_points(:,1), stable_points(:,2), 'ro', ...
     'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);

% Mark unstable point (saddle point at origin)
plot(0, 0, 'yo', 'MarkerSize', 10, 'MarkerFaceColor', 'y', 'LineWidth', 2);

xlabel('x', 'FontSize', 12);
ylabel('y', 'FontSize', 12);
title('(b) The vertical view of w(x,y)', 'FontSize', 14);
axis equal;
xlim([-6, 6]);
ylim([-6, 6]);

sgtitle(sprintf('The graph of w(x,y) and its projection onto the x-y plane\n(a=%.1f, b=%.1f, c=%.1f, d=%.1f, r=%.1f)', ...
        a, b, c, d, r), 'FontSize', 16, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'TDCBSR_potential_function.png');
fprintf('Potential function visualization saved to TDCBSR_potential_function.png\n');

%% Display system information
fprintf('\n========================================\n');
fprintf('TDCBSR Potential Function\n');
fprintf('========================================\n');
fprintf('Parameters:\n');
fprintf('  a = %.2f (controlled system parameter)\n', a);
fprintf('  b = %.2f (controlled system parameter)\n', b);
fprintf('  c = %.2f (controlling system parameter)\n', c);
fprintf('  d = %.2f (controlling system parameter)\n', d);
fprintf('  r = %.2f (coupling coefficient)\n', r);
fprintf('\nPotential function:\n');
fprintf('  w(x,y) = (b/4)*y^4 + (d/4)*x^4 - (1/2)*(c+r)*x^2 - (1/2)*(r+a)*y^2 + r*x*y\n');
fprintf('\nThe system has:\n');
fprintf('  - 4 stable points (potential wells)\n');
fprintf('  - 1 unstable point (saddle point at origin)\n');
fprintf('========================================\n');
