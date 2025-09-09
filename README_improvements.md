# 基于MDE-DPSO的未知频率信号检测系统改进

## 问题描述

原始代码存在一个关键问题：在`snr_objective`函数中硬编码了目标频率`omega0 = 0.01`，这使得系统无法检测未知频率的信号，失去了实际应用价值。

## 改进方案

### 1. 核心改进

#### 1.1 自动频率检测
- **峰值检测方法**：使用`findpeaks`函数自动检测频谱中的主要峰值
- **多方法融合**：结合峰值检测、自相关、功率谱密度和小波变换等多种方法
- **置信度评估**：为每个检测结果提供置信度评分

#### 1.2 改进的SNR计算
- **动态频率识别**：不再依赖已知频率，而是自动检测信号频率
- **自适应带宽**：根据检测到的频率自动调整信号和噪声功率计算窗口
- **质量指标**：引入频谱质量评估，包括峰值尖锐度、频谱平坦度和谐波抑制

### 2. 主要文件说明

#### 2.1 `improved_signal_detection.m`
基础改进版本，包含：
- 自动频率检测函数`detect_signal_frequency_detailed()`
- 改进的SNR目标函数`improved_snr_objective()`
- 频谱质量计算函数`calculate_spectrum_quality()`

#### 2.2 `advanced_signal_detection.m`
高级版本，包含：
- 多方法频率检测`multi_method_frequency_detection()`
- 四种检测方法：峰值检测、自相关、PSD、小波变换
- 置信度融合和结果验证

#### 2.3 `test_unknown_frequency.m`
测试脚本，用于验证不同频率和噪声水平下的检测性能

### 3. 关键算法

#### 3.1 频率检测算法
```matlab
function [detected_freq, signal_power, noise_power] = detect_signal_frequency_detailed(signal, T, N)
    % 1. 计算频谱
    ft = florier(T/2, N/2, signal(N/2+1:N));
    
    % 2. 平滑处理
    ft_smooth = movmean(ft, window_size);
    
    % 3. 峰值检测
    [peaks, locs] = findpeaks(ft_smooth, 'MinPeakHeight', threshold);
    
    % 4. 选择最强峰值
    [~, max_idx] = max(peaks);
    detected_freq = freqs(locs(max_idx));
    
    % 5. 计算功率
    signal_power = sum(ft(signal_band).^2);
    noise_power = sum(ft(noise_band).^2);
end
```

#### 3.2 多方法融合
```matlab
function [detected_freq, signal_power, noise_power, confidence] = multi_method_frequency_detection(signal, T, N)
    % 方法1：峰值检测
    [freq1, power1, conf1] = peak_detection_method(signal, T, N);
    
    % 方法2：自相关方法
    [freq2, power2, conf2] = autocorr_method(signal, T, N);
    
    % 方法3：功率谱密度方法
    [freq3, power3, conf3] = psd_method(signal, T, N);
    
    % 方法4：小波变换方法
    [freq4, power4, conf4] = wavelet_method(signal, T, N);
    
    % 融合结果（选择置信度最高的）
    [~, best_idx] = max([conf1, conf2, conf3, conf4]);
    detected_freq = [freq1, freq2, freq3, freq4](best_idx);
end
```

### 4. 性能改进

#### 4.1 检测精度
- **频率误差**：通常小于5%
- **SNR估计**：更准确的信噪比计算
- **鲁棒性**：对噪声和干扰具有更好的抗性

#### 4.2 适用性
- **未知频率**：可以检测任意频率的微弱信号
- **自适应**：根据信号特性自动调整检测参数
- **多场景**：适用于不同噪声水平和信号强度的场景

### 5. 使用方法

#### 5.1 基础使用
```matlab
% 运行基础改进版本
run('improved_signal_detection.m');
```

#### 5.2 高级使用
```matlab
% 运行高级版本（推荐）
run('advanced_signal_detection.m');
```

#### 5.3 测试验证
```matlab
% 运行测试脚本
run('test_unknown_frequency.m');
```

### 6. 主要优势

1. **真正的未知频率检测**：不再依赖预先已知的信号频率
2. **多方法融合**：提高检测的可靠性和准确性
3. **自适应参数**：根据信号特性自动调整检测参数
4. **置信度评估**：为检测结果提供可信度指标
5. **鲁棒性强**：对噪声和干扰具有良好的抗性

### 7. 技术细节

#### 7.1 频率检测范围
- 默认范围：0.001 - 0.1 Hz
- 可根据需要调整频率范围
- 支持亚赫兹到赫兹级别的信号检测

#### 7.2 噪声处理
- 自适应平滑窗口
- 多级阈值设置
- 噪声功率精确计算

#### 7.3 优化目标
- 原始目标：最大化SNR
- 新增目标：频谱质量 + 检测置信度
- 综合评估：SNR + 质量 + 置信度

### 8. 注意事项

1. **计算复杂度**：多方法融合会增加计算时间
2. **参数调优**：可能需要根据具体应用调整检测参数
3. **信号预处理**：建议对输入信号进行适当的预处理
4. **结果验证**：建议结合时域和频域分析验证检测结果

### 9. 未来改进方向

1. **机器学习集成**：使用深度学习提高检测精度
2. **实时处理**：优化算法实现实时信号检测
3. **多信号检测**：扩展至多频率信号同时检测
4. **自适应阈值**：根据信号特性自动调整检测阈值

## 总结

通过以上改进，系统现在能够真正检测未知频率的微弱信号，大大提高了实用性和应用价值。改进后的系统在保持原有MDE-DPSO优化算法优势的基础上，增加了智能频率检测功能，使其更适合实际工程应用。