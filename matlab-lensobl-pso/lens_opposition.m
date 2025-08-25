function opp_positions = lens_opposition(positions, lower_bound, upper_bound)
% Apply a simple lens opposition transformation for each particle
% The lens center is the midpoint between bounds; we refract positions through it
num_particles = size(positions, 1);
num_dimensions = size(positions, 2);
center = (upper_bound + lower_bound) / 2;

% Lens factor randomly sampled in (0.5, 1.5) per particle-dimension
k = 0.5 + rand(num_particles, num_dimensions);

% Opposite through lens: c +/- k*(c - x)
opp_positions = center + k .* (center - positions);

% Clip to bounds
opp_positions = max(opp_positions, lower_bound);
opp_positions = min(opp_positions, upper_bound);
end
