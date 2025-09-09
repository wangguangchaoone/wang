clc; clear; close all;
global k kexi zuni

PopSize = 50;
Max_Iter = 50;     
Dim = 3;           
LB = [0.1, 0.1, 0.7]; 
UB = [5.0, 5.0, 2.0];  

% 信号及参数初始化
jili = 0.1; v = 0.01; set_noise = 0.6;
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
[Gbest, gbestValue, curve] = MDE_DPSO(@(x) improved_snr_objective(x,signal_add_noise), ...
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

% 检测到的频率
[detected_freq, detected_snr] = detect_signal_frequency(x_1, T, N);
fprintf('检测到的信号频率: %.4f Hz\n', detected_freq);
fprintf('检测到的SNR: %.4f dB\n', detected_snr);

% 绘图
figure(1)
subplot(2,1,1)
plot(t(1:chang)/t_per, x1(1:chang), 'k', 'LineWidth', 1.5); grid on;
xlabel({'周期/个';'(b) 输出信号时域图'})
ylabel('幅值/V');

subplot(2,1,2)
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

figure(2)
subplot(2,1,1)
plot(t0(1:chang)/t_per,signal_add_noise(1:chang),'k','LineWidth',1.5);hold on;
grid on; ylabel('Amplitude/V'); xlabel('Number of periods');

subplot(2,1,2)
plot((4*pi/T)*[0:199],ft_0(1:200),'k','LineWidth',1.5);
xlabel('Frequency/Hz'); ylabel('Spectrum Amplitude/V');

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

function snr_val = improved_snr_objective(params, signal_add_noise)
% 改进的SNR目标函数，自动检测信号频率
% params: [a, b, zuni]
if nargin < 2
    error('improved_snr_objective: 输入参数不足。用法: improved_snr_objective([a,b,zuni], signal_add_noise)');
end
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
t0 = 0:dt:T-dt;

% Duffing系统响应
f0 = [0];
[~, x1] = runge_kutta_mn(@test_fun3, f0, dt, 0, T-dt, signal_add_noise);
x_1 = x1';

% 检查输出信号是否异常
if any(isnan(x_1)) || any(isinf(x_1))
    fprintf('Warning: Duffing system output contains NaN/Inf\n');
    snr_val = -100;
    return;
end

% 计算频谱
ft_1 = florier(N/2*dt, N/2, x_1(N/2+1:N));

% 检查频谱是否异常
if any(isnan(ft_1)) || any(isinf(ft_1))
    fprintf('Warning: Spectrum contains NaN/Inf\n');
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

% 调试信息（低概率打印）
if rand < 0.05
    fprintf(['Debug - Params[%.2f,%.2f,%.2f]: Freq=%.4f Hz, SNR=%.2f dB, ', ...
             'Quality=%.2f, obj=%.2f\n'], ...
            k, kexi, zuni, detected_freq, output_snr, spectrum_quality, objective);
end

% 返回：最小化问题取相反数
snr_val = -objective;
end

function [detected_freq, signal_power, noise_power] = detect_signal_frequency_detailed(signal, T, N)
% 详细的信号频率检测函数
% 输入：signal - 时域信号, T - 总时间, N - 采样点数
% 输出：detected_freq - 检测到的频率, signal_power - 信号功率, noise_power - 噪声功率

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
% 考虑：1) 峰值尖锐度 2) 频谱平坦度 3) 谐波抑制

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

% 四阶龙格库塔数值积分
function [x,y1]=runge_kutta_mn(ufunc,y0,h,a,b,noise)
% 输入参数：ufunc微分方程函数，y0初始值，h步长，a,b时间区间，noise噪声信号

n=floor((b-a)/h);% 步数
x(1)=a;% 时间起点
y(:,1)=y0;% 初始值，注意维度匹配

for ii=1:n
    x(ii+1)=x(ii)+h;
    
    % 计算k1, k2, k3, k4
    k1=ufunc(x(ii),y(:,ii),noise(ii));
    k2=ufunc(x(ii)+h/2,y(:,ii)+h*k1/2,noise(ii));
    k3=ufunc(x(ii)+h/2,y(:,ii)+h*k2/2,noise(ii));
    k4=ufunc(x(ii)+h,y(:,ii)+h*k3,noise(ii));
    
    % 检查k值是否异常
    if any(isnan([k1;k2;k3;k4])) || any(isinf([k1;k2;k3;k4]))
        fprintf('Warning: RK4 coefficients contain NaN/Inf at step %d\n', ii);
        y(:,ii+1) = y(:,ii); % 保持上一步的值
        continue;
    end
    
    % 计算下一步
    y(:,ii+1)=y(:,ii)+h*(k1+2*k2+2*k3+k4)/6;
    
    % 检查输出是否异常
    if any(isnan(y(:,ii+1))) || any(isinf(y(:,ii+1)))
        fprintf('Warning: RK4 output contains NaN/Inf at step %d\n', ii);
        y(:,ii+1) = y(:,ii); % 保持上一步的值
    end
    
    % 限制输出范围，防止数值溢出
    y(:,ii+1) = max(-1e6, min(1e6, y(:,ii+1)));
end

y1=y';
end

% Duffing系统微分方程
function dy=test_fun3(t,x,signal)

global k;    % 线性恢复系数
global zuni; % 阻尼系数
global kexi; % 非线性系数

% 检查输入参数是否异常
if any(isnan(x)) || any(isinf(x)) || any(isnan(signal)) || any(isinf(signal))
    dy = [0]; % 返回零，避免数值发散
    return;
end

% 计算Duffing系统响应
dx_dt = (k*x(1) - kexi*x(1)^3 + signal) / zuni;

% 检查输出是否异常
if isnan(dx_dt) || isinf(dx_dt)
    dy = [0]; % 返回零，避免数值发散
    return;
end

% 限制输出范围，防止数值溢出
if abs(dx_dt) > 1e6
    dx_dt = sign(dx_dt) * 1e6;
end

dy = [dx_dt];
end

function ft = florier(T1,N,signal)
% 离散傅里叶变换
% T1 : 时间窗口长度
% N  : 频率点数
% signal: 输入信号

% 检查输入参数
if isempty(signal) || any(isnan(signal)) || any(isinf(signal))
    ft = zeros(1,N);
    return;
end

T0=T1;             %时间窗口
T = T0/N;           %采样时间间隔
t = [0:1:N-1]*T;   %时间向量
span_t=length(t);      %离散点的长度

% 检查信号长度
if length(signal) < span_t
    ft = zeros(1,N);
    return;
end

x=signal; 

w_c=zeros(N,span_t);    %准备cos的N维投影矩阵
w_s=zeros(N,span_t);    %准备sin的N维投影矩阵
for k=0:N-1    % 取频率为N个，那么离散频谱的采样频率为 2*pi/T0
    w_c(k+1,:)=cos(k*2*pi/T0*t);  %cos的每个频率的离散时间投影
    w_s(k+1,:)=sin(k*2*pi/T0*t);  %sin的每个频率的离散时间投影
end

c_c=zeros(1,N);    %准备投影系数向量
c_s=zeros(1,N);    

for n=1:N
    c_c(n)=(x*w_c(n,:)')/(N);     %将信号投影到每个频率的cos投影上
    c_s(n)=(x*w_s(n,:)')/(N);     %将信号投影到每个频率的sin投影上
end

% 检查投影系数是否异常
c_c(isnan(c_c)) = 0;
c_c(isinf(c_c)) = 0;
c_s(isnan(c_s)) = 0;
c_s(isinf(c_s)) = 0;

for n=1:N
    ft(n)=((c_c(n))^2+(c_s(n))^2)^0.5;   %将cos投影系数和sin投影系数合成
end

% 检查最终结果
ft(isnan(ft)) = 0;
ft(isinf(ft)) = 0;
end