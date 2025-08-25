

function [result, history] = pso_lensobl(objective, lower_bound, upper_bound, params)
% PSO with Lens Opposition-Based Learning (Lens-OBL) and Selective Randomization (SR)
% objective: handle f(x)
% lower_bound, upper_bound: 1xD vectors
% params: struct of algorithm parameters

% Dimensions and swarm size
num_dimensions = numel(lower_bound);
num_particles = params.numParticles;
max_iterations = params.maxIterations;

% Bound ranges and velocity limits
range_vec = upper_bound - lower_bound;
v_max = params.vMaxFrac * range_vec;

% Initialize swarm positions and velocities
positions = lower_bound + rand(num_particles, num_dimensions) .* range_vec;
velocities = zeros(num_particles, num_dimensions);

% Evaluate initial fitness
fitness = arrayfun(@(i) objective(positions(i, :)), 1:num_particles)';

% Personal bests
pbest_positions = positions;
pbest_fitness = fitness;

% Global best
[gbest_fitness, best_idx] = min(pbest_fitness);
gbest_position = pbest_positions(best_idx, :);

% History
history.bestFitness = zeros(max_iterations, 1);

% Stagnation tracker
no_improve_counter = 0;
prev_best = gbest_fitness;

for iter = 1:max_iterations
	% inertia weight schedule
	w = params.wMax - (params.wMax - params.wMin) * (iter - 1) / (max_iterations - 1);

	% Lens-OBL candidate generation for current global best neighborhood
	if params.enableLensOBL
		% create opposition population around each particle's personal best using lens transformation
		opp_positions = lens_opposition(pbest_positions, lower_bound, upper_bound);
		% Evaluate and greedily select better between current and opposite for each particle
		opp_fitness = arrayfun(@(i) objective(opp_positions(i, :)), 1:num_particles)';
		improved_mask = opp_fitness < pbest_fitness;
		pbest_positions(improved_mask, :) = opp_positions(improved_mask, :);
		pbest_fitness(improved_mask) = opp_fitness(improved_mask);
		% update global best possibly via OBL
		[local_best_fit, local_best_idx] = min(pbest_fitness);
		if local_best_fit < gbest_fitness
			gbest_fitness = local_best_fit;
			gbest_position = pbest_positions(local_best_idx, :);
		end
	end

	% Update velocities and positions
	[velocities, positions] = update_velocity_position(positions, velocities, pbest_positions, gbest_position, w, params.c1, params.c2, v_max, lower_bound, upper_bound);

	% Evaluate swarm
	fitness = arrayfun(@(i) objective(positions(i, :)), 1:num_particles)';

	% SR trigger conditions (sigma >> A or l << 1 or a too large)
	if params.enableSR
		% sigma: swarm positional std averaged over dims
		sigma = mean(std(positions, 0, 1));
		A = mean(range_vec);
		% l: normalized improvement in this iter
		improvement = prev_best - gbest_fitness;
		l = improvement / max(1e-12, abs(prev_best));
		% a: average velocity magnitude relative to range
		a = mean(vecnorm(velocities, 2, 2)) / (norm(range_vec) / sqrt(num_dimensions));

		cond_large_sigma = sigma > params.largeSigmaFactor * (A / 10);
		cond_small_l = l < params.smallLThreshold * 1e-3; % very small relative improvement
		cond_large_a = a > params.largeAThreshold;

		if cond_large_sigma || cond_small_l || cond_large_a
			% Apply Selective Randomization around global best
			positions = sr_operator(positions, gbest_position, lower_bound, upper_bound, 0.25);
			% Re-evaluate fitness after SR
			fitness = arrayfun(@(i) objective(positions(i, :)), 1:num_particles)';
		end
	end

	% Update personal and global bests
	is_better = fitness < pbest_fitness;
	pbest_positions(is_better, :) = positions(is_better, :);
	pbest_fitness(is_better) = fitness(is_better);

	[new_best, idx] = min(pbest_fitness);
	if new_best < gbest_fitness
		gbest_fitness = new_best;
		gbest_position = pbest_positions(idx, :);
	end

	% Stagnation tracking
	if gbest_fitness < prev_best - 1e-12
		no_improve_counter = 0;
		prev_best = gbest_fitness;
	else
		no_improve_counter = no_improve_counter + 1;
	end

	% Optional SR by stagnation counter
	if params.enableSR && no_improve_counter >= params.stagnationIter
		positions = sr_operator(positions, gbest_position, lower_bound, upper_bound, 0.5);
		no_improve_counter = 0;
		fitness = arrayfun(@(i) objective(positions(i, :)), 1:num_particles)';
	end

	history.bestFitness(iter) = gbest_fitness;
end

result.bestFitness = gbest_fitness;
result.bestPosition = gbest_position;
result.iterations = max_iterations;
end
