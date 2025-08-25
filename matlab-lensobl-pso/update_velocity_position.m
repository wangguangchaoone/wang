function [velocities, positions] = update_velocity_position(positions, velocities, pbest_positions, gbest_position, w, c1, c2, v_max, lower_bound, upper_bound)
num_particles = size(positions, 1);
num_dimensions = size(positions, 2);

r1 = rand(num_particles, num_dimensions);
r2 = rand(num_particles, num_dimensions);

velocities = w .* velocities ...
	+ c1 .* r1 .* (pbest_positions - positions) ...
	+ c2 .* r2 .* (gbest_position - positions);

% clamp velocity per dimension
velocities = max(min(velocities, v_max), -v_max);

positions = positions + velocities;

% clamp positions
positions = max(positions, lower_bound);
positions = min(positions, upper_bound);
end
