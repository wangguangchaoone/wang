function [best_params, best_fitness, convergence] = LensOBL_PSO(fitness_func, dim, lb, ub, max_iter, pop_size, signal_freq)
% LensOBL改进的粒子群算法

%% 初始化PSO参数
w_max = 0.9;  w_min = 0.4;  c1 = 2.0;  c2 = 2.0;

%% 初始化种群
X = lb + (ub - lb) .* rand(pop_size, dim);
V = zeros(pop_size, dim);

fitness = zeros(pop_size, 1);
for i = 1:pop_size
    fitness(i) = fitness_func(X(i, :));
end

pbest = X;
pbest_fitness = fitness;
[gbest_fitness, idx] = max(fitness);
gbest = X(idx, :);
convergence = zeros(max_iter, 1);

%% 主循环
for iter = 1:max_iter
    w = w_max - (w_max - w_min) * iter / max_iter;
    w_lens = w_max - (w_max - w_min) / max_iter * iter;
    
    for i = 1:pop_size
        V(i, :) = w * V(i, :) + c1 * rand(1, dim) .* (pbest(i, :) - X(i, :)) + c2 * rand(1, dim) .* (gbest - X(i, :));
        V(i, :) = max(min(V(i, :), 0.2 * (ub - lb)), -0.2 * (ub - lb));
        X(i, :) = X(i, :) + V(i, :);
        X(i, :) = max(min(X(i, :), ub), lb);
        
        if rand < 0.5
            X_obl = lensOBL(X(i, :), lb, ub, w_lens);
            X_obl = max(min(X_obl, ub), lb);
            fitness_obl = fitness_func(X_obl);
            fitness_current = fitness_func(X(i, :));
            if fitness_obl > fitness_current
                X(i, :) = X_obl;
                fitness(i) = fitness_obl;
            else
                fitness(i) = fitness_current;
            end
        else
            fitness(i) = fitness_func(X(i, :));
        end
        
        if fitness(i) > pbest_fitness(i)
            pbest(i, :) = X(i, :);
            pbest_fitness(i) = fitness(i);
        end
    end
    
    [current_best, idx] = max(fitness);
    if current_best > gbest_fitness
        gbest_fitness = current_best;
        gbest = X(idx, :);
    end
    
    convergence(iter) = gbest_fitness;
    if mod(iter, 10) == 0
        fprintf('迭代 %d/%d, 最优MSNR = %.4f\n', iter, max_iter, gbest_fitness);
    end
end

best_params = gbest;
best_fitness = gbest_fitness;
end

function X_obl = lensOBL(X, lb, ub, w_lens)
p = lb; q = ub; dim = length(X); X_obl = zeros(1, dim);
for d = 1:dim
    r = (p(d) + q(d) - 2*X(d)) / (2*(w_lens + 1));
    X_obl(d) = (1 + 1/w_lens) * r + (p(d) + q(d)) / 2;
end
end
