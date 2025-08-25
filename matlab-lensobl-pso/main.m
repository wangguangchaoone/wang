% Main script for Lens-OBL based PSO with SR
% Choose objective function and bounds
objective.name = 'ackley'; % options: 'ackley', 'sphere'
switch objective.name
	case 'ackley'
		objective.func = @benchmark_ackley;
		objective.bounds = [-32.768, 32.768];
	case 'sphere'
		objective.func = @benchmark_sphere;
		objective.bounds = [-100, 100];
	otherwise
		error('Unknown objective');
end

% Problem dimension and bounds
D = 30;
lower_bound = objective.bounds(1) * ones(1, D);
upper_bound = objective.bounds(2) * ones(1, D);

% PSO parameters
params.numParticles = 40;
params.maxIterations = 500;
params.c1 = 2.05; % cognitive
params.c2 = 2.05; % social
params.wMax = 0.9; % inertia weight range
params.wMin = 0.4;
params.vMaxFrac = 0.2; % v_max as fraction of range

% Lens-OBL and SR settings
params.enableLensOBL = true;
params.enableSR = true;      % selective randomization when stagnation or anomalies
params.stagnationIter = 20;  % t threshold for SR
params.largeSigmaFactor = 10;% rule: sigma >> A
params.smallLThreshold = 1;  % rule: l << 1
params.largeAThreshold = 1.0;% rule: a too large (step amplitude)

rng('default');

[result, history] = pso_lensobl(objective.func, lower_bound, upper_bound, params);

fprintf('Best fitness: %.6g\n', result.bestFitness);
fprintf('Iterations: %d\n', result.iterations);

utils_plot(history.bestFitness, 'Best fitness over iterations');
