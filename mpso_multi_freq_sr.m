clc; clear; close all;
global k kexi zuni
% 说明：该函数在信噪比计算中已经优化（加入了在保留SNR的基础上引入"全频段显著性"与"竞争峰惩罚"）
% 和mde-dpso相差的地方只有算法上不同
% 多频信号版本：使用MSNR（平均信噪比）作为优化目标

PopSize = 50;      % 增加粒子数，提高多样性
Max_Iter = 20;     % 增加迭代次数，充分收敛
Dim = 3;           % 优化参数个数[a, b, zuni]3
LB = [0.1, 0.1, 0.7]; % 排除极差参数区域
UB = [5.0, 5.0, 2.0];  % 集中在有效参数区间

% 信号及参数初始化
jili = 0.1;
v1 = 0.01;   % 第一个频率
v2 = 0.045;  % 第二个频率
freqs = [v1, v2];  % 频率数组，用于传递给适应度函数
set_noise = 0.6;
% zuni = 4; % 
n1 = 8;
n2 = 1000;
t_per = 2*pi/v1;  % 使用第一个频率作为周期参考
dt = t_per/n2;
N = n1*n2;
T = N*dt;
t0 = 0:dt:T-dt;

% 多频信号生成
signal = sin(v1*t0) + sin(v2*t0);  % 多频信号

chang = length(t0);
noise = randn(1, chang);
signal_set = jili * signal;
signal_add_noise = set_noise * noise + signal_set;

% 运行MPSO，传递signal_add_noise和频率数组
[Gbest, gbestValue, ~, ~, MSNR_history] = MPSO(PopSize, Dim, Max_Iter, 0, 1e-8, LB, UB, 0, 1, signal_add_noise, freqs);

fprintf('最优参数: a=%.4f, b=%.4f, zuni=%.4f\n', Gbest(1), Gbest(2), Gbest(3)); 
fprintf('最优MSNR: %.4f dB\n', -gbestValue);

% 用最优参数画出系统响应和频谱
k = Gbest(1); kexi = Gbest(2); zuni = Gbest(3);

f0 = [0];
[t, x1] = runge_kutta_mn(@test_fun3, f0, dt, 0, T-dt, signal_add_noise);
x_1 = x1';
ft_1 = florier(N/2*dt, N/2, x_1(N/2+1:N));
x_0 = signal_add_noise;
ft_0 = florier(N/2*dt, N/2, x_0(N/2+1:N));

figure(1)
subplot(2,1,1)
plot(t(1:chang)/t_per, x1(1:chang), 'k'); grid on;%, 'LineWidth', 1.5
xlabel({'周期/个';'(b) 输出信号时域图'})
ylabel('幅值/V');

subplot(2,1,2)
plot((4*pi/T)*(0:199), ft_1(1:200), 'k');hold on;%, 'LineWidth', 1.5
grid on;
xlabel({'频率/Hz';'(b) 输出信号频谱'})
ylabel('频谱幅值/V');
datacursormode(gcf,'on');
set(findall(gcf,'ViewStyle','datatip'),'fontsize',56) ;

figure(2)%0.1    1
        subplot(2,1,1)

        plot(t0(1:chang)/t_per,signal_add_noise(1:chang),'k');hold on;  % , 'LineWidth', 1.5第一个参数：周期个数；第二个参数：加噪信号；k:代表线的颜色。hold on:保持当前图形，允许后续的 plot 命令在同一图中叠加绘制
        grid on;  % 在图中显示网格线，便于观察数据点的数值。

        ylabel(['Amplitude/V']);
      xlabel(['Number of periods'])

         subplot(2,1,2)
        plot((4*pi/T)*[0:199],ft_0(1:200),'k');  % , 'LineWidth', 1.5第一个参数：取前200频点；第二个参数：前200个幅值；

        % xlabel({'频率/Hz';'(b) 系统输入信号频谱图'})
        xlabel(['Frequency/Hz'])
        ylabel(['Spectrum Amplitude/V']); 
        datacursormode(gcf,'on');
    set(findall(gcf,'ViewStyle','datatip'),'fontsize',56) ;

% 绘制MSNR随迭代次数变化的图
figure(3)
plot(0:Max_Iter, MSNR_history, 'b-', 'LineWidth', 1);
grid on;
xlabel('迭代次数');
ylabel('MSNR (dB)');
title('MSNR随迭代次数的变化');
legend('MSNR', 'Location', 'best');

% 添加最终MSNR值的标注
hold on;
plot(Max_Iter, MSNR_history(end), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(Max_Iter, MSNR_history(end), sprintf('  最终MSNR: %.2f dB', MSNR_history(end)), ...
     'FontSize', 10, 'Color', 'red');
hold off;

% 显示收敛信息
fprintf('初始MSNR: %.4f dB\n', MSNR_history(1));
fprintf('最终MSNR: %.4f dB\n', MSNR_history(end));
fprintf('MSNR提升: %.4f dB\n', MSNR_history(end) - MSNR_history(1));

disp('优化与仿真完成！'); 
%================================================
% MPSO: A Modified Particle Swarm Optimization Using Adaptive Strategy
% % Authors: Hao Liu, XueWei Zhang, LiangPing Tu
% Journal: Expert System With Applications
% Email: haoliu@ustl.edu.cn,    liuhustl@sina.cn
%================================================
function [Gbest, gbestValue, FEs, Fbest, MSNR_history]=MPSO(N,dim,T_Max,fun,err,LB,UB,opt_f,run,signal_add_noise,freqs)
Vmax0=0.5*(UB-LB); % 0.5*(UB-LB)
%% Inertial Weight
% w=0.72984;
w_max=0.9;%初始化权重因子
w_min=0.4;
c_max = 2.5;%初始化学习因子2.5改（1）
c_min = 0.5;
Fbest=[];   
MSNR_history = [];  % 记录每次迭代的MSNR值
x=0.72;  %rand
%% Learning Factor
c1=2;%1.49618;2 改（2）
c2=c1;
%% Initialize Population
Xmax=repmat(UB,N,1);
Xmin=repmat(LB,N,1); %绘制了一个可寻找空间
X=Xmin+(Xmax-Xmin).*rand(N,dim); %生成了N个随机粒子
Vmax=repmat(Vmax0,N,1); % 初始速度
V=-Vmax+2*Vmax.*rand(N,dim); % 初始速度
%% Evaluation of particles.
fX(N)=inf;
A = 0.1; % SR判别阈值，可根据实际调整
for i=1:N
    a = X(i,1); % 假设a为第1维
    b = X(i,2); % 假设b为第2维
    % 计算SR参数
    sigma = sqrt(4*a^3/(27*b));
    K = 2*a;
    l = 2*sqrt(a/b);
    % SR判别
    if (sigma > 10*A) || (l < 0.01) || (a > 5) % 判别条件可根据实际调整
        fX(i) = 1e6; % 适应度极差 
    else
        fX(i) = msnr_objective(X(i,:), signal_add_noise, freqs);
    end
end
FEs=N;
%% Initialize Pbest and Gbest
Pbest=X;   
fPbest=fX;
[gbestValue, gbestIndex]=min(fPbest); 
Gbest=Pbest(gbestIndex,:); 
Fbest=[Fbest gbestValue];
% 记录初始MSNR值（转换为正值）
MSNR_history = [MSNR_history, -gbestValue];   
%% Iteration
for t=1:T_Max
    %% Update Position X and Velocity V and check the responding bounds
    x=4*x*(1-x);
    w=x*w_min+(w_max-w_min)*t/T_Max;
    %  c1,c2学习因子调整新加
    kk = t;
    G = T_Max;
    c1 = c_min + (c_max - c_min) * exp(-(4*kk/G)^2);
    c2 = c_max - (c_max - c_min) * exp(-(4*kk/G)^2);
    
    % 自适应参数调整：在后期增加局部搜索
    if t > T_Max * 0.7
        c1 = c1 * 0.8;  % 减少个体学习，增加局部搜索
        c2 = c2 * 1.2;  % 增加社会学习，向全局最优收敛
    end
    
    % 精英保留：保留最优的10%粒子
    elite_size = max(1, round(N * 0.1));
    [~, elite_idx] = sort(fPbest);
    elite_idx = elite_idx(1:elite_size);
    
    % 标准PSO速度更新
    V = w*V + c1*rand(N,dim).*(Pbest - X) + c2*rand(N,dim).*(repmat(Gbest,N,1) - X);
    V = max(-Vmax, min(Vmax, V));
    
    % 位置更新（精英粒子保持不动）
    for i = 1:N
        if ~ismember(i, elite_idx)  % 非精英粒子才更新
            X(i,:) = X(i,:) + V(i,:);
        end
    end
    X = max(Xmin, min(Xmax, X));
    %% 
    for i=1:N
        a = X(i,1); b = X(i,2);
        sigma = sqrt(4*a^3/(27*b));
        l = 2*sqrt(a/b);
        if (sigma > 10*A) || (l < 0.01) || (a > 5)
            fX(i) = 1e6;
        else
            fX(i) = msnr_objective(X(i,:), signal_add_noise, freqs);
        end
    end
    
    %=== LensOBL镜像学习机制 ===
    w_max = 2.0;
    w_min = 0.5;
    % w_t = w_max - (w_max - w_min) * t / T_Max;线性变为非线性
    w_t = w_max - (w_max - w_min) * cos((pi/2)*(t / T_Max));
    for i = 1:N
        for d = 1:dim 
            p = LB(d);
            q = UB(d);
            x = X(i,d);
            r = (p + q - 2*x) / (2 * (w_t + 1));
            x_star = (1 + 1/w_t) * r + (p + q)/2 - x/w_t;
            x_star = max(p, min(q, x_star));
            X_mirror(i,d) = x_star;
        end
        a_m = X_mirror(i,1); b_m = X_mirror(i,2);
        sigma_m = sqrt(4*a_m^3/(27*b_m));
        l_m = 2*sqrt(a_m/b_m);
        if (sigma_m > 10*A) || (l_m < 0.01) || (a_m > 5)
            fX_mirror = 1e6;
        else
            fX_mirror = msnr_objective(X_mirror(i,:), signal_add_noise, freqs);
        end
        if fX_mirror < fX(i)
            X(i,:) = X_mirror(i,:);
            fX(i) = fX_mirror;
        end
    end
    %% ==Evaluate Fitness and Change Pbest
    for i=1:N
        if fX(i) < fPbest(i)
            Pbest(i,:)=X(i,:);
            fPbest(i)=fX(i);
        end
        if fPbest(i)<gbestValue
            Gbest=Pbest(i,:);
            gbestValue=fPbest(i);
        end
    end
    FEs = FEs + N;
    %% Record GbestValue and MSNR for plotting
    MSNR_history = [MSNR_history, -gbestValue];  % 记录每次迭代的MSNR值（转换为正值）
     if rem(t,4)==0
         Fbest=[Fbest gbestValue];
     end  
end % end iteration
end % end function    

%% MSNR目标函数（多频信号的平均信噪比）
function msnr_val = msnr_objective(params, signal_add_noise, freqs)
% params: [a, b, zuni]
% signal_add_noise: 输入的加噪信号
% freqs: 频率数组，例如 [0.01, 0.045]
if nargin < 3
    error('msnr_objective: 输入参数不足。用法: msnr_objective([a,b,zuni], signal_add_noise, freqs)');
end
global k kexi zuni
k = params(1);    % a
kexi = params(2); % b
zuni = params(3); % zuni

% 直接使用传入的signal_add_noise
n1 = 8;
n2 = 1000;
t_per = 2*pi/freqs(1);  % 使用第一个频率作为周期参考
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
    fprintf('Warning: Duffing system output contains NaN/Inf ');
    msnr_val = -100;
    return;
end

% 计算频谱
ft_1 = florier(N/2*dt, N/2, x_1(N/2+1:N));

% 检查频谱是否异常
if any(isnan(ft_1)) || any(isinf(ft_1))
    fprintf('Warning: Spectrum contains NaN/Inf ');
    msnr_val = -100;
    return;
end

% 对每个频率计算SNR，然后取平均得到MSNR
num_freqs = length(freqs);
snr_values = zeros(1, num_freqs);

for freq_idx = 1:num_freqs
    omega = freqs(freq_idx);  % 当前角频率 rad/s
    
    % 计算主频索引（florier 的频率步长为 4*pi/T，单位为rad/s）
    signal_idx = round((omega * T) / (4*pi)) + 1;
    
    % 检查索引是否有效
    if signal_idx < 1 || signal_idx > length(ft_1)
        fprintf('Warning: Invalid signal index: %d for freq %.4f ', signal_idx, omega);
        snr_values(freq_idx) = -100;
        continue;
    end
    
    % 计算信号功率
    output_signal_power = ft_1(signal_idx)^2;
    if isnan(output_signal_power) || isinf(output_signal_power)
        fprintf('Warning: Signal power is NaN/Inf for freq %.4f ', omega);
        snr_values(freq_idx) = -100;
        continue;
    end
    
    % 计算噪声功率
    output_noise_power = sum(ft_1.^2) - output_signal_power + eps;
    if isnan(output_noise_power) || isinf(output_noise_power) || output_noise_power <= 0
        fprintf('Warning: Noise power is invalid for freq %.4f ', omega);
        snr_values(freq_idx) = -100;
        continue;
    end
    
    % 计算SNR（dB）
    output_snr = 10*log10(output_signal_power / output_noise_power);
    
    % 最终检查
    if isnan(output_snr) || isinf(output_snr)
        fprintf('Warning: Final SNR is NaN/Inf for freq %.4f ', omega);
        snr_values(freq_idx) = -100;
        continue;
    end
    
    %% --- 在保留SNR的基础上引入"全频段显著性"与"竞争峰惩罚" ---
    % 1) 显著性：目标峰与其余频点（排除若干谐波邻域）的分离度
    % 2) 竞争峰惩罚：若任何非谐波峰高于目标峰则惩罚
    
    L = length(ft_1);
    target_amp = ft_1(signal_idx);
    amp_db = 20*log10(ft_1 + eps);
    target_db = 20*log10(target_amp + eps);
    
    % 排除目标及前K阶谐波的小邻域
    K_harm = 5;      % 考虑前5阶谐波
    bw = 4;          % 每个谐波周围排除的bin半宽
    mask = true(1, L);
    for kk = 1:K_harm
        idx = kk * signal_idx;
        if idx < 1 || idx > L, break; end
        left  = max(1, idx - bw);
        right = min(L, idx + bw);
        mask(left:right) = false;
    end
    % 同时排除直流与边界
    mask(1:max(2,bw)) = false;
    mask(end-bw+1:end) = false;
    
    competitor_db = amp_db(mask);
    if isempty(competitor_db)
        competitor_db = amp_db; % 兜底
    end
    
    % 使用Top-K竞争峰均值作为参考，强调与"最强对手群"的分离
    K = 5;  % Top-K
    sorted_comp = sort(competitor_db, 'descend');
    topk_mean_db = mean(sorted_comp(1:min(K, numel(sorted_comp))));
    separation_db = target_db - topk_mean_db;  % 目标相对最强对手群的优势
    
    % 竞争峰惩罚（弱化）：仅当最强对手超过目标时才惩罚
    max_competitor_db = sorted_comp(1);
    exceed_db = max(0, max_competitor_db - target_db);
    
    % 组合目标：越大越好
    %   objective = alpha*SNR + beta*separation - gamma*exceed
    alpha = 1.0; beta = 1.2; gamma = 0.6;  % 更偏向分离度，减小惩罚过度
    objective = alpha*output_snr + beta*separation_db - gamma*exceed_db;
    
    snr_values(freq_idx) = objective;
end

% 计算MSNR（平均信噪比）
msnr = mean(snr_values);

% 调试信息（低概率打印）
if rand < 0.05
    fprintf(['Debug - Params[%.2f,%.2f,%.2f]: SNRs=['], ...
            k, kexi, zuni);
    for i = 1:num_freqs
        fprintf('%.2f', snr_values(i));
        if i < num_freqs
            fprintf(', ');
        end
    end
    fprintf('], MSNR=%.2f dB\n', msnr);
end

% 返回：最小化问题取相反数
msnr_val = -msnr;
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
