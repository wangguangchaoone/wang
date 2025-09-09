clc; clear; close all;
global k kexi zuni

PopSize = 50;
Max_Iter = 50;     
Dim = 3;           
LB = [0.1, 0.1, 0.7]; 
UB = [5.0, 5.0, 2.0];  

% 信号及参数初始化 - 使用未知频率
% 随机生成测试频率（模拟未知信号）
unknown_freq = 0.005 + rand() * 0.02; % 0.005-0.025 Hz范围内的随机频率
fprintf('真实信号频率: %.4f Hz (系统未知)\n', unknown_freq);

jili = 0.1; v = unknown_freq; set_noise = 0.6;
n1 = 8; n2 = 1000;
t_per = 2*pi/v; dt = t_per/n2;
N = n1*n2; T = N*dt;
t0 = 0:dt:T-dt;
signal = sin(v*t0);
chang = length(t0);
noise = randn(1, chang);
signal_set = jili * signal;
signal_add_noise = set_noise * noise + signal_set;

% ============= 运行改进的 MDE-DPSO =================
[Gbest, gbestValue, curve] = MDE_DPSO(@(x) advanced_snr_objective(x,signal_add_noise), ...
                                      Dim, PopSize, Max_Iter, LB, UB);

fprintf('最优参数: a=%.4f, b=%.4f, zuni=%.4f\n',  Gbest(1), Gbest(2),Gbest(3));
fprintf('最优SNR: %.4f dB\n', -gbestValue);

% 设置最优参数
k = Gbest(1); kexi = Gbest(2); zuni = Gbest(3);

% Duffing系统输出
f0 = [0];
[t, x1] = runge_kutta_mn(@test_fun3, f0, dt, 0, T-dt, signal_add_noise);
x_1 = x1';
ft_1 = florier(N/2*dt, N/2, x_1(N/2+1:N));
x_0 = signal_add_noise;
ft_0 = florier(N/2*dt, N/2, x_0(N/2+1:N));

% 高级频率检测
[detected_freq, detected_snr, confidence] = advanced_frequency_detection(x_1, T, N);
fprintf('检测到的信号频率: %.4f Hz\n', detected_freq);
fprintf('检测到的SNR: %.4f dB\n', detected_snr);
fprintf('检测置信度: %.2f%%\n', confidence * 100);
fprintf('频率检测误差: %.2f%%\n', abs(detected_freq - unknown_freq) / unknown_freq * 100);

% 绘图
figure(1)
subplot(3,1,1)
plot(t(1:chang)/t_per, x1(1:chang), 'k', 'LineWidth', 1.5); grid on;
xlabel({'周期/个';'(a) Duffing系统输出信号'})
ylabel('幅值/V');
title(sprintf('真实频率: %.4f Hz, 检测频率: %.4f Hz', unknown_freq, detected_freq));

subplot(3,1,2)
plot((4*pi/T)*(0:199), ft_1(1:200), 'k', 'LineWidth', 1.5);hold on;
% 标记检测到的频率
detected_idx = round((detected_freq * T) / (4*pi)) + 1;
if detected_idx <= length(ft_1)
    plot((4*pi/T)*(detected_idx-1), ft_1(detected_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text((4*pi/T)*(detected_idx-1), ft_1(detected_idx), sprintf('%.4f Hz', detected_freq), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end
grid on;
xlabel({'频率/Hz';'(b) 输出信号频谱'})
ylabel('频谱幅值/V');

subplot(3,1,3)
plot(t0(1:chang)/t_per,signal_add_noise(1:chang),'k','LineWidth',1.5);hold on;
grid on; ylabel('Amplitude/V'); xlabel('Number of periods');
title('输入信号（含噪声）');

figure(2)
plot((4*pi/T)*[0:199],ft_0(1:200),'k','LineWidth',1.5);
xlabel('Frequency/Hz'); ylabel('Spectrum Amplitude/V');
title('输入信号频谱');
hold on;
% 标记真实频率
real_idx = round((unknown_freq * T) / (4*pi)) + 1;
if real_idx <= length(ft_0)
    plot((4*pi/T)*(real_idx-1), ft_0(real_idx), 'go', 'MarkerSize', 8, 'LineWidth', 2);
    text((4*pi/T)*(real_idx-1), ft_0(real_idx), sprintf('真实: %.4f Hz', unknown_freq), ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'Color', 'green');
end

disp('优化与仿真完成！');

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

function snr_val = advanced_snr_objective(params, signal_add_noise)
% 高级SNR目标函数，使用多种频率检测方法
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

% 使用多种方法检测频率
[detected_freq, signal_power, noise_power, confidence] = multi_method_frequency_detection(x_1, T, N);

if detected_freq <= 0 || signal_power <= 0 || noise_power <= 0 || confidence < 0.3
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

% 计算综合质量指标
quality = calculate_advanced_quality(ft_1, detected_freq, T, confidence);

% 组合目标函数：SNR + 质量 + 置信度
alpha = 1.0; beta = 0.3; gamma = 0.2;
objective = alpha * output_snr + beta * quality + gamma * confidence * 20;

% 调试信息（低概率打印）
if rand < 0.05
    fprintf(['Debug - Params[%.2f,%.2f,%.2f]: Freq=%.4f Hz, SNR=%.2f dB, ', ...
             'Quality=%.2f, Conf=%.2f, obj=%.2f\n'], ...
            k, kexi, zuni, detected_freq, output_snr, quality, confidence, objective);
end

% 返回：最小化问题取相反数
snr_val = -objective;
end

function [detected_freq, detected_snr, confidence] = advanced_frequency_detection(signal, T, N)
% 高级频率检测函数，结合多种方法
[detected_freq, signal_power, noise_power, confidence] = multi_method_frequency_detection(signal, T, N);

if detected_freq > 0 && signal_power > 0 && noise_power > 0
    detected_snr = 10*log10(signal_power / noise_power);
else
    detected_snr = -100;
end
end

function [detected_freq, signal_power, noise_power, confidence] = multi_method_frequency_detection(signal, T, N)
% 多方法频率检测
% 方法1：峰值检测
[freq1, power1, conf1] = peak_detection_method(signal, T, N);

% 方法2：自相关方法
[freq2, power2, conf2] = autocorr_method(signal, T, N);

% 方法3：功率谱密度方法
[freq3, power3, conf3] = psd_method(signal, T, N);

% 方法4：小波变换方法
[freq4, power4, conf4] = wavelet_method(signal, T, N);

% 融合结果
candidates = [freq1, freq2, freq3, freq4];
confidences = [conf1, conf2, conf3, conf4];
powers = [power1, power2, power3, power4];

% 选择置信度最高的结果
[~, best_idx] = max(confidences);
detected_freq = candidates(best_idx);
signal_power = powers(best_idx);

% 计算噪声功率
ft = florier(T/2, N/2, signal(N/2+1:N));
df = 1/T;
freqs = (0:N/2-1) * df;
[~, signal_idx] = min(abs(freqs - detected_freq));
bw = max(1, round(0.002/df));
left_idx = max(1, signal_idx - bw);
right_idx = min(length(ft), signal_idx + bw);
noise_mask = true(1, length(ft));
noise_mask(left_idx:right_idx) = false;
noise_power = sum(ft(noise_mask).^2) + eps;

% 综合置信度
confidence = mean(confidences);
end

function [freq, power, conf] = peak_detection_method(signal, T, N)
% 峰值检测方法
ft = florier(T/2, N/2, signal(N/2+1:N));
df = 1/T;
freqs = (0:N/2-1) * df;

% 平滑频谱
window_size = max(3, round(0.01/df));
if window_size > 1
    ft_smooth = movmean(ft, window_size);
else
    ft_smooth = ft;
end

% 寻找峰值
[peaks, locs] = findpeaks(ft_smooth, 'MinPeakHeight', max(ft_smooth)*0.1, ...
                          'MinPeakDistance', round(0.005/df));

if isempty(peaks)
    freq = 0; power = 0; conf = 0;
    return;
end

% 选择最强的峰值
[~, max_idx] = max(peaks);
freq = freqs(locs(max_idx));
power = peaks(max_idx)^2;

% 计算置信度（基于峰值与噪声的比值）
noise_level = median(ft_smooth);
conf = min(1, peaks(max_idx) / (noise_level + eps));
end

function [freq, power, conf] = autocorr_method(signal, T, N)
% 自相关方法
% 计算自相关函数
autocorr_signal = xcorr(signal, 'normalized');
autocorr_signal = autocorr_signal(N:end); % 只取正延迟部分

% 寻找自相关的峰值（排除零延迟）
[peaks, locs] = findpeaks(autocorr_signal(2:end), 'MinPeakHeight', 0.3);
locs = locs + 1; % 调整索引

if isempty(peaks)
    freq = 0; power = 0; conf = 0;
    return;
end

% 选择最强的峰值
[~, max_idx] = max(peaks);
period_samples = locs(max_idx);
freq = 1 / (period_samples * T / N);
power = peaks(max_idx)^2;
conf = min(1, peaks(max_idx));
end

function [freq, power, conf] = psd_method(signal, T, N)
% 功率谱密度方法
% 使用Welch方法估计PSD
window = hann(min(256, length(signal)));
noverlap = round(length(window)/2);
nfft = 2^nextpow2(length(signal));

[psd, f] = pwelch(signal, window, noverlap, nfft, 1/T);

% 寻找PSD峰值
[peaks, locs] = findpeaks(psd, 'MinPeakHeight', max(psd)*0.1);

if isempty(peaks)
    freq = 0; power = 0; conf = 0;
    return;
end

% 选择最强的峰值
[~, max_idx] = max(peaks);
freq = f(locs(max_idx));
power = peaks(max_idx);
conf = min(1, peaks(max_idx) / (mean(psd) + eps));
end

function [freq, power, conf] = wavelet_method(signal, T, N)
% 小波变换方法（简化版）
% 使用Morlet小波进行连续小波变换
dt = T / N;
t = 0:dt:(N-1)*dt;

% 定义频率范围
freqs = 0.001:0.001:0.05; % 1-50 mHz
scales = 1 ./ freqs;

% 简化的连续小波变换
cwt_result = zeros(length(scales), N);
for i = 1:length(scales)
    scale = scales(i);
    % 简化的Morlet小波
    sigma = scale / 5;
    wavelet = exp(-(t - N*dt/2).^2 / (2*sigma^2)) .* exp(1i * 2*pi * freqs(i) * t);
    cwt_result(i, :) = abs(conv(signal, conj(fliplr(wavelet)), 'same'));
end

% 寻找最大能量
[~, max_scale_idx] = max(max(cwt_result, [], 2));
freq = freqs(max_scale_idx);
power = max(cwt_result(max_scale_idx, :))^2;

% 计算置信度
energy_ratio = max(cwt_result(max_scale_idx, :)) / (mean(cwt_result(:)) + eps);
conf = min(1, energy_ratio);
end

function quality = calculate_advanced_quality(ft, detected_freq, T, confidence)
% 计算高级质量指标
N = length(ft);
df = 1/T;
freqs = (0:N-1) * df;

% 找到检测频率对应的索引
[~, signal_idx] = min(abs(freqs - detected_freq));

% 1. 峰值尖锐度
bw = max(1, round(0.002/df));
left_idx = max(1, signal_idx - bw);
right_idx = min(N, signal_idx + bw);
peak_region = ft(left_idx:right_idx);
peak_value = max(peak_region);
neighbor_avg = mean(peak_region(peak_region < peak_value));
sharpness = peak_value / (neighbor_avg + eps);

% 2. 频谱平坦度
spectrum_flatness = geomean(ft + eps) / (mean(ft) + eps);

% 3. 谐波抑制
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

% 4. 频率稳定性（基于置信度）
stability = confidence;

% 组合质量指标
quality = sharpness * spectrum_flatness * stability - harmonic_penalty;
quality = max(0, quality);
end

% 其他辅助函数保持不变
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