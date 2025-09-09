% 测试未知频率检测系统
clc; clear; close all;

% 测试不同频率的信号
test_frequencies = [0.005, 0.01, 0.015, 0.02, 0.025];
noise_levels = [0.3, 0.6, 1.0];

fprintf('=== 未知频率检测系统测试 ===\n\n');

for i = 1:length(test_frequencies)
    for j = 1:length(noise_levels)
        fprintf('测试频率: %.3f Hz, 噪声水平: %.1f\n', test_frequencies(i), noise_levels(j));
        
        % 生成测试信号
        v = test_frequencies(i);
        jili = 0.1;
        set_noise = noise_levels(j);
        n1 = 8; n2 = 1000;
        t_per = 2*pi/v;
        dt = t_per/n2;
        N = n1*n2;
        T = N*dt;
        t0 = 0:dt:T-dt;
        signal = sin(v*t0);
        chang = length(t0);
        noise = randn(1, chang);
        signal_set = jili * signal;
        signal_add_noise = set_noise * noise + signal_set;
        
        % 运行改进的检测系统
        try
            [Gbest, gbestValue, curve] = MDE_DPSO(@(x) improved_snr_objective(x,signal_add_noise), ...
                                                  3, 30, 20, [0.1, 0.1, 0.7], [5.0, 5.0, 2.0]);
            
            % 检测频率
            [detected_freq, detected_snr] = detect_signal_frequency(signal_add_noise, T, N);
            
            % 计算检测误差
            freq_error = abs(detected_freq - v) / v * 100;
            
            fprintf('  真实频率: %.4f Hz\n', v);
            fprintf('  检测频率: %.4f Hz\n', detected_freq);
            fprintf('  频率误差: %.2f%%\n', freq_error);
            fprintf('  检测SNR: %.2f dB\n', detected_snr);
            fprintf('  优化SNR: %.2f dB\n', -gbestValue);
            fprintf('  最优参数: [%.3f, %.3f, %.3f]\n\n', Gbest(1), Gbest(2), Gbest(3));
            
        catch ME
            fprintf('  错误: %s\n\n', ME.message);
        end
    end
end

% 包含所有必要的函数定义
function snr_val = improved_snr_objective(params, signal_add_noise)
% 改进的SNR目标函数，自动检测信号频率
global k kexi zuni
k = params(1);    % a
kexi = params(2); % b
zuni = params(3); % zuni

% 直接使用传入的signal_add_noise
n1 = 8;
n2 = 1000;
t_per = 2*pi/0.01;
dt = t_per/n2;
N = n1*n2;
T = N*dt;

% Duffing系统响应
f0 = [0];
[~, x1] = runge_kutta_mn(@test_fun3, f0, dt, 0, T-dt, signal_add_noise);
x_1 = x1';

% 检查输出信号是否异常
if any(isnan(x_1)) || any(isinf(x_1))
    snr_val = -100;
    return;
end

% 计算频谱
ft_1 = florier(N/2*dt, N/2, x_1(N/2+1:N));

% 检查频谱是否异常
if any(isnan(ft_1)) || any(isinf(ft_1))
    snr_val = -100;
    return;
end

% 自动检测信号频率
[detected_freq, signal_power, noise_power] = detect_signal_frequency_detailed(x_1, T, N);

if detected_freq <= 0 || signal_power <= 0 || noise_power <= 0
    snr_val = -100;
    return;
end

% 计算SNR（dB）
output_snr = 10*log10(signal_power / noise_power);

% 最终检查
if isnan(output_snr) || isinf(output_snr)
    snr_val = -100;
    return;
end

% 计算频谱质量指标
spectrum_quality = calculate_spectrum_quality(ft_1, detected_freq, T);

% 组合目标函数：SNR + 频谱质量
alpha = 1.0; beta = 0.5;
objective = alpha * output_snr + beta * spectrum_quality;

% 返回：最小化问题取相反数
snr_val = -objective;
end

function [detected_freq, signal_power, noise_power] = detect_signal_frequency_detailed(signal, T, N)
% 详细的信号频率检测函数
% 计算频谱
ft = florier(T/2, N/2, signal(N/2+1:N));

% 频率分辨率
df = 1/T;  % Hz
freqs = (0:N/2-1) * df;

% 平滑频谱以减少噪声影响
window_size = max(3, round(0.01/df)); % 自适应窗口大小
if window_size > 1
    ft_smooth = movmean(ft, window_size);
else
    ft_smooth = ft;
end

% 寻找峰值
[peaks, locs] = findpeaks(ft_smooth, 'MinPeakHeight', max(ft_smooth)*0.1, ...
                          'MinPeakDistance', round(0.005/df));

if isempty(peaks)
    detected_freq = 0;
    signal_power = 0;
    noise_power = sum(ft.^2);
    return;
end

% 选择最强的峰值作为信号频率
[~, max_idx] = max(peaks);
signal_idx = locs(max_idx);
detected_freq = freqs(signal_idx);

% 计算信号功率（峰值周围的小窗口）
bw = max(1, round(0.002/df)); % 信号带宽
left_idx = max(1, signal_idx - bw);
right_idx = min(length(ft), signal_idx + bw);
signal_power = sum(ft(left_idx:right_idx).^2);

% 计算噪声功率（排除信号频带）
noise_mask = true(1, length(ft));
noise_mask(left_idx:right_idx) = false;
noise_power = sum(ft(noise_mask).^2) + eps;

% 验证检测结果
if detected_freq < 0.001 || detected_freq > 0.1 || signal_power < noise_power * 0.1
    detected_freq = 0;
    signal_power = 0;
    noise_power = sum(ft.^2);
end
end

function [detected_freq, detected_snr] = detect_signal_frequency(signal, T, N)
% 简化的信号频率检测函数（用于主程序显示）
[detected_freq, signal_power, noise_power] = detect_signal_frequency_detailed(signal, T, N);

if detected_freq > 0 && signal_power > 0 && noise_power > 0
    detected_snr = 10*log10(signal_power / noise_power);
else
    detected_snr = -100;
end
end

function quality = calculate_spectrum_quality(ft, detected_freq, T)
% 计算频谱质量指标
N = length(ft);
df = 1/T;
freqs = (0:N-1) * df;

% 找到检测频率对应的索引
[~, signal_idx] = min(abs(freqs - detected_freq));

% 1. 峰值尖锐度：峰值与邻域的比例
bw = max(1, round(0.002/df));
left_idx = max(1, signal_idx - bw);
right_idx = min(N, signal_idx + bw);
peak_region = ft(left_idx:right_idx);
peak_value = max(peak_region);
neighbor_avg = mean(peak_region(peak_region < peak_value));
sharpness = peak_value / (neighbor_avg + eps);

% 2. 频谱平坦度：整体频谱的平坦程度
spectrum_flatness = geomean(ft + eps) / (mean(ft) + eps);

% 3. 谐波抑制：检查是否有明显的谐波
harmonic_penalty = 0;
for h = 2:5
    harmonic_idx = round(h * signal_idx);
    if harmonic_idx <= N && harmonic_idx > 0
        harmonic_ratio = ft(harmonic_idx) / (peak_value + eps);
        if harmonic_ratio > 0.3
            harmonic_penalty = harmonic_penalty + harmonic_ratio;
        end
    end
end

% 组合质量指标
quality = sharpness * spectrum_flatness - harmonic_penalty;
quality = max(0, quality); % 确保非负
end

function [gbest, gbest_fitness, curve] = MDE_DPSO(fhd, dim, N, MaxIter, lb, ub)
    % 参数
    wmax = 0.9; wmin = 0.4;
    F = 0.5; CR = 0.9;  
    p = 0.2;            
    Nar = 50;           
    Perturb_strength = 0.01;

    % 初始化
    X = rand(N, dim) .* (ub - lb) + lb;
    V = zeros(N, dim);
    fitness = arrayfun(@(i) fhd(X(i,:)), 1:N)';
    pbest = X; pbest_fit = fitness;
    [gbest_fitness, idx] = min(fitness);
    gbest = X(idx,:);
    archive = repmat(gbest, Nar, 1);
    unimproved = zeros(N,1);
    curve = zeros(MaxIter,1);

    for t = 1:MaxIter
        % 动态参数
        w  = wmin + (wmax-wmin) * cos(pi/2 * t/MaxIter);
        log_term = log(1 + (t/MaxIter)*(exp(1)-1));
        c1 = 2 * sin((1-log_term) * pi/2);
        c2 = 2 * cos((1-log_term) * pi/2);
        c3 = 1.0;

        % 群体中心最近粒子
        Center = mean(X,1);
        [~, idxCNP] = min(vecnorm(X - Center,2,2));
        CNP = X(idxCNP,:);

        % 更新速度和位置
        for i = 1:N
            r1 = rand; r2 = rand; r3 = rand;
            Perturb = Perturb_strength*(rand(1,dim)-0.5);
            V(i,:) = w*V(i,:) + c1*r1*(pbest(i,:) - X(i,:)) + c2*r2*(gbest - X(i,:)) + c3*r3*(CNP - X(i,:)) + Perturb;
            X(i,:) = X(i,:) + V(i,:);
        end
        X = min(max(X, lb), ub);

        % DE 变异交叉选择
        for i = 1:N
            if unimproved(i) >= 10
                % DE/rand/1
                idxs = randperm(N,3);
                Vi = X(idxs(1),:) + F*(X(idxs(2),:) - X(idxs(3),:));
            else
                % DE/current-to-pbest/1
                [~, sortIdx] = sort(pbest_fit);
                topk = sortIdx(1:max(1,round(p*N)));
                xpbest = pbest(topk(randi(numel(topk))),:);
                Xr4 = X(randi(N),:);
                if rand < 0.5
                    Xr5 = archive(randi(size(archive,1)),:);
                else
                    Xr5 = X(randi(N),:);
                end
                Vi = X(i,:) + F*(xpbest - X(i,:)) + F*(Xr4 - Xr5);
            end

            Ui = X(i,:);
            jrand = randi(dim);
            for j = 1:dim
                if rand < CR || j == jrand
                    Ui(j) = Vi(j);
                end
            end

            fitU = fhd(Ui);
            if fitU < fitness(i)
                X(i,:) = Ui; fitness(i) = fitU; unimproved(i) = 0;
                if fitU < pbest_fit(i)
                    pbest(i,:) = Ui; pbest_fit(i) = fitU;
                end
                if fitU < gbest_fitness
                    gbest = Ui; gbest_fitness = fitU;
                end
                % 更新档案
                archive(randi(size(archive,1)),:) = Ui;
            else
                unimproved(i) = unimproved(i) + 1;
            end
        end
        curve(t) = gbest_fitness;
    end
end

function [x,y1]=runge_kutta_mn(ufunc,y0,h,a,b,noise)
n=floor((b-a)/h);
x(1)=a;
y(:,1)=y0;

for ii=1:n
    x(ii+1)=x(ii)+h;
    k1=ufunc(x(ii),y(:,ii),noise(ii));
    k2=ufunc(x(ii)+h/2,y(:,ii)+h*k1/2,noise(ii));
    k3=ufunc(x(ii)+h/2,y(:,ii)+h*k2/2,noise(ii));
    k4=ufunc(x(ii)+h,y(:,ii)+h*k3,noise(ii));
    
    if any(isnan([k1;k2;k3;k4])) || any(isinf([k1;k2;k3;k4]))
        y(:,ii+1) = y(:,ii);
        continue;
    end
    
    y(:,ii+1)=y(:,ii)+h*(k1+2*k2+2*k3+k4)/6;
    
    if any(isnan(y(:,ii+1))) || any(isinf(y(:,ii+1)))
        y(:,ii+1) = y(:,ii);
    end
    
    y(:,ii+1) = max(-1e6, min(1e6, y(:,ii+1)));
end

y1=y';
end

function dy=test_fun3(t,x,signal)
global k zuni kexi

if any(isnan(x)) || any(isinf(x)) || any(isnan(signal)) || any(isinf(signal))
    dy = [0];
    return;
end

dx_dt = (k*x(1) - kexi*x(1)^3 + signal) / zuni;

if isnan(dx_dt) || isinf(dx_dt)
    dy = [0];
    return;
end

if abs(dx_dt) > 1e6
    dx_dt = sign(dx_dt) * 1e6;
end

dy = [dx_dt];
end

function ft = florier(T1,N,signal)
if isempty(signal) || any(isnan(signal)) || any(isinf(signal))
    ft = zeros(1,N);
    return;
end

T0=T1;
T = T0/N;
t = [0:1:N-1]*T;
span_t=length(t);

if length(signal) < span_t
    ft = zeros(1,N);
    return;
end

x=signal; 

w_c=zeros(N,span_t);
w_s=zeros(N,span_t);
for k=0:N-1
    w_c(k+1,:)=cos(k*2*pi/T0*t);
    w_s(k+1,:)=sin(k*2*pi/T0*t);
end

c_c=zeros(1,N);
c_s=zeros(1,N);    

for n=1:N
    c_c(n)=(x*w_c(n,:)')/(N);
    c_s(n)=(x*w_s(n,:)')/(N);
end

c_c(isnan(c_c)) = 0;
c_c(isinf(c_c)) = 0;
c_s(isnan(c_s)) = 0;
c_s(isinf(c_s)) = 0;

for n=1:N
    ft(n)=((c_c(n))^2+(c_s(n))^2)^0.5;
end

ft(isnan(ft)) = 0;
ft(isinf(ft)) = 0;
end