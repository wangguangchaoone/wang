function [best_params, best_SNR] = BSO_optimization(input_signal, target_freqs, fs, h, ...
                                                      param_ranges, max_iter, pop_size, n_clusters)
% BSO_optimization - Brain Storm Optimization algorithm for TDCBSR parameters
%
% Optimizes parameters a, b, c, d, r to maximize output SNR
%
% Inputs:
%   input_signal - Input signal with noise
%   target_freqs - Target signal frequencies (Hz)
%   fs - Sampling frequency
%   h - Time step for RK4
%   param_ranges - Structure with fields: a, b, c, d, r (each [min, max])
%   max_iter - Maximum number of iterations
%   pop_size - Population size
%   n_clusters - Number of clusters for k-means
%
% Outputs:
%   best_params - Best parameters [a, b, c, d, r]
%   best_SNR - Best SNR achieved

% Initialize population randomly
n_params = 5;  % a, b, c, d, r
population = zeros(pop_size, n_params);
fitness = zeros(pop_size, 1);

% Parameter order: [a, b, c, d, r]
fields = {'a', 'b', 'c', 'd', 'r'};
for i = 1:pop_size
    for j = 1:n_params
        field = fields{j};
        population(i, j) = param_ranges.(field)(1) + ...
                          (param_ranges.(field)(2) - param_ranges.(field)(1)) * rand();
    end
end

% Evaluate initial population
for i = 1:pop_size
    fitness(i) = evaluate_fitness(population(i, :), input_signal, target_freqs, fs, h);
end

% BSO parameters
p_replace = 0.2;  % Probability to replace cluster center
p_one = 0.4;      % Probability to select one cluster
p_one_center = 0.4;  % Probability to select cluster center
p_two = 0.5;      % Probability to select two clusters

% Main BSO loop
for iter = 1:max_iter
    % K-means clustering
    if pop_size >= n_clusters
        [idx, centers] = kmeans(population, n_clusters, 'MaxIter', 100);
    else
        idx = (1:pop_size)';
        centers = population;
        n_clusters = pop_size;
    end
    
    % Calculate fitness for cluster centers
    center_fitness = zeros(n_clusters, 1);
    for i = 1:n_clusters
        center_fitness(i) = evaluate_fitness(centers(i, :), input_signal, target_freqs, fs, h);
    end
    
    % Mutation and Update
    for i = 1:pop_size
        new_individual = population(i, :);
        
        if rand() < p_one
            % Select one cluster
            cluster_id = randi(n_clusters);
            
            if rand() < p_one_center
                % Select cluster center
                new_individual = centers(cluster_id, :);
            else
                % Select random individual from cluster
                cluster_members = find(idx == cluster_id);
                if ~isempty(cluster_members)
                    selected = cluster_members(randi(length(cluster_members)));
                    new_individual = population(selected, :);
                end
            end
        else
            % Select two clusters
            cluster_ids = randperm(n_clusters, min(2, n_clusters));
            
            if rand() < p_two && length(cluster_ids) == 2
                % Combine two cluster centers
                new_individual = (centers(cluster_ids(1), :) + centers(cluster_ids(2), :)) / 2;
            else
                % Select random individuals from two clusters
                if length(cluster_ids) >= 2
                    cluster_members1 = find(idx == cluster_ids(1));
                    cluster_members2 = find(idx == cluster_ids(2));
                    if ~isempty(cluster_members1) && ~isempty(cluster_members2)
                        ind1 = cluster_members1(randi(length(cluster_members1)));
                        ind2 = cluster_members2(randi(length(cluster_members2)));
                        new_individual = (population(ind1, :) + population(ind2, :)) / 2;
                    end
                end
            end
        end
        
        % Add Gaussian mutation
        mutation = randn(1, n_params) * 0.1 * (iter/max_iter);  % Decreasing mutation
        new_individual = new_individual + mutation;
        
        % Ensure parameters are within bounds
        for j = 1:n_params
            field = fields{j};
            new_individual(j) = max(param_ranges.(field)(1), ...
                                   min(param_ranges.(field)(2), new_individual(j)));
        end
        
        % Evaluate new individual
        new_fitness = evaluate_fitness(new_individual, input_signal, target_freqs, fs, h);
        
        % Update if better
        if new_fitness > fitness(i)
            population(i, :) = new_individual;
            fitness(i) = new_fitness;
        end
    end
    
    % Display progress
    [best_SNR, best_idx] = max(fitness);
    if mod(iter, 10) == 0
        fprintf('Iteration %d/%d: Best SNR = %.2f dB\n', iter, max_iter, best_SNR);
    end
end

% Return best solution
[best_SNR, best_idx] = max(fitness);
best_params = population(best_idx, :);

fprintf('Optimization completed!\n');
fprintf('Best parameters: a=%.4f, b=%.4f, c=%.4f, d=%.4f, r=%.4f\n', ...
        best_params(1), best_params(2), best_params(3), best_params(4), best_params(5));
fprintf('Best SNR: %.2f dB\n', best_SNR);

end

function fitness = evaluate_fitness(params, input_signal, target_freqs, fs, h)
% Evaluate fitness (SNR) for given parameters
    a = params(1);
    b = params(2);
    c = params(3);
    d = params(4);
    r = params(5);
    
    % Run TDCBSR system
    try
        [y, ~] = TDCBSR_RK4(input_signal, a, b, c, d, r, h);
        
        % Calculate SNR (Equation 8)
        fitness = calculate_SNR(y, target_freqs, fs);
        
        % Penalize if SNR is too low or NaN
        if isnan(fitness) || isinf(fitness)
            fitness = -1000;
        end
    catch
        fitness = -1000;
    end
end
