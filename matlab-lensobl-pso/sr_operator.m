function new_positions = sr_operator(positions, gbest_position, lower_bound, upper_bound, radius_frac)
% Selective Randomization: re-sample subset of particles within a radius around gbest
num_particles = size(positions, 1);
num_dimensions = size(positions, 2);

range_vec = upper_bound - lower_bound;
radius = radius_frac * range_vec; % per-dimension radius

% Randomly choose half particles to diversify
mask = rand(num_particles, 1) < 0.5;
new_positions = positions;

if any(mask)
	noise = (rand(sum(mask), num_dimensions) - 0.5) .* (2 .* radius);
	cand = gbest_position + noise;
	cand = max(cand, lower_bound);
	cand = min(cand, upper_bound);
	new_positions(mask, :) = cand;
end
end
