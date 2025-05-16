% filepath: g:\zizim\Documents\code\matlab_project\demo_2\audio_processing.m
% 音频处理工具函数

function [varargout] = audio_processing(operation, varargin)
    % 音频处理主函数，作为工具函数的接口
    % operation: 操作类型，如 'addWhiteNoise', 'addNarrowbandNoise', 等
    % varargin: 根据操作类型传递给各子函数的参数
    % varargout: 根据操作类型返回的输出参数
    
    % 确定要返回的输出参数数量
    num_outputs = nargout;
    
    % 根据操作类型调用相应的子函数
    switch operation
        case 'addWhiteNoise'
            out = cell(1, num_outputs);
            [out{:}] = addWhiteNoise(varargin{:});
        case 'addNarrowbandNoise'
            out = cell(1, num_outputs);
            [out{:}] = addNarrowbandNoise(varargin{:});
        case 'addSinusoidalNoise'
            out = cell(1, num_outputs);
            [out{:}] = addSinusoidalNoise(varargin{:});
        case 'applyFIRFilter'
            out = cell(1, num_outputs);
            [out{:}] = applyFIRFilter(varargin{:});
        case 'getFilterResponse'
            out = cell(1, max(num_outputs, 2)); % 至少需要2个输出
            [out{:}] = getFilterResponse(varargin{:});
        case 'applyLMSFilter'
            out = cell(1, num_outputs);
            [out{:}] = applyLMSFilter(varargin{:});
        case 'applyWaveletDenoising'
            out = cell(1, num_outputs);
            [out{:}] = applyWaveletDenoising(varargin{:});
        case 'applyNotchFilter'
            out = cell(1, num_outputs);
            [out{:}] = applyNotchFilter(varargin{:});
        otherwise
            error('未知的操作类型: %s', operation);
    end
    
    % 将子函数的输出赋值给输出参数
    for i = 1:num_outputs
        varargout{i} = out{i};
    end
end

function noisy = addWhiteNoise(signal, SNR_dB)
    % 向信号添加高斯白噪声，SNR指定
    % SNR_dB: 信噪比（分贝）
    
    signal_power = mean(signal.^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    noise = sqrt(noise_power) * randn(size(signal));
    noisy = signal + noise;
    
    % 归一化以防止削波
    if max(abs(noisy)) > 1
        noisy = noisy / max(abs(noisy));
    end
end

function noisy = addNarrowbandNoise(signal, fs, f_low, f_high, SNR_dB)
    % 在[f_low, f_high]Hz范围内添加窄带高斯噪声
    % fs: 采样频率
    % f_low, f_high: 低频和高频边界(Hz)
    % SNR_dB: 信噪比（分贝）
    
    N = length(signal);
    
    % 生成白噪声
    white_noise = randn(size(signal));
    
    % 设计带通滤波器
    nyquist = fs/2;
    [b, a] = butter(4, [f_low/nyquist, f_high/nyquist], 'bandpass');
    
    % 应用滤波器得到窄带噪声
    narrowband_noise = filter(b, a, white_noise);
    
    % 调整噪声功率以达到所需SNR
    signal_power = mean(signal.^2);
    noise_power_current = mean(narrowband_noise.^2);
    noise_power_desired = signal_power / (10^(SNR_dB/10));
    
    scale = sqrt(noise_power_desired / noise_power_current);
    narrowband_noise = scale * narrowband_noise;
    
    % 将噪声添加到信号
    noisy = signal + narrowband_noise;
    
    % 归一化以防止削波
    if max(abs(noisy)) > 1
        noisy = noisy / max(abs(noisy));
    end
end

function noisy = addSinusoidalNoise(signal, fs, freq, amplitude)
    % 添加指定频率的正弦干扰
    % fs: 采样频率
    % freq: 干扰频率(Hz)
    % amplitude: 干扰幅度
    
    N = length(signal);
    t = (0:N-1)/fs;
    
    % 生成正弦噪声
    sine_noise = amplitude * sin(2*pi*freq*t)';
    
    % 添加到信号
    noisy = signal + sine_noise;
    
    % 归一化以防止削波
    if max(abs(noisy)) > 1
        noisy = noisy / max(abs(noisy));
    end
end

function filtered = applyFIRFilter(signal, fs, type, cutoff, window_type)
    % 使用窗函数法应用FIR滤波器
    % type: 'low'(低通), 'high'(高通), 或 'stop'(带阻)
    % cutoff: 低通/高通的截止频率，或带阻的[低频 高频]
    % window_type: '巴特利特窗', '汉宁窗', '汉明窗', '布莱克曼窗', 或 '凯泽窗'
    
    % 滤波器阶数（带阻滤波器应为偶数）
    order = 100;
    
    % 创建窗
    switch window_type
        case '巴特利特窗'
            win = bartlett(order+1);
        case '汉宁窗'
            win = hann(order+1);
        case '汉明窗'
            win = hamming(order+1);
        case '布莱克曼窗'
            win = blackman(order+1);
        case '凯泽窗'
            win = kaiser(order+1, 5); % Beta = 5
    end
    
    % 设计滤波器
    nyquist = fs/2;
    switch type
        case 'low'
            cutoff_norm = cutoff/nyquist;
            b = fir1(order, cutoff_norm, 'low', win);
        case 'high'
            cutoff_norm = cutoff/nyquist;
            b = fir1(order, cutoff_norm, 'high', win);
        case 'stop'
            cutoff_norm = cutoff/nyquist;
            b = fir1(order, cutoff_norm, 'stop', win);
    end
    
    % 应用滤波器
    filtered = filter(b, 1, signal);
end

function [h, w] = getFilterResponse(type, cutoff, fs, window_type)
    % 计算滤波器频率响应
    % 返回幅度响应h和频率w
    
    order = 100;
    
    % 创建窗
    switch window_type
        case '巴特利特窗'
            win = bartlett(order+1);
        case '汉宁窗'
            win = hann(order+1);
        case '汉明窗'
            win = hamming(order+1);
        case '布莱克曼窗'
            win = blackman(order+1);
        case '凯泽窗'
            win = kaiser(order+1, 5); % Beta = 5
    end
    
    % 设计滤波器
    nyquist = fs/2;
    switch type
        case 'low'
            cutoff_norm = cutoff/nyquist;
            b = fir1(order, cutoff_norm, 'low', win);
        case 'high'
            cutoff_norm = cutoff/nyquist;
            b = fir1(order, cutoff_norm, 'high', win);
        case 'stop'
            cutoff_norm = cutoff/nyquist;
            b = fir1(order, cutoff_norm, 'stop', win);
    end
    
    % 计算频率响应
    [h, w] = freqz(b, 1, 1024);
end

function filtered = applyLMSFilter(noisy, desired, mu, filterOrder)
    % 应用自定义LMS自适应滤波器，基于博客实现
    % noisy: 带噪声信号
    % desired: 期望信号（若为空，则自动生成参考信号）
    % mu: 步长参数（若为空，则使用默认值0.01）
    % filterOrder: 滤波器长度（若为空，则使用默认值32）
    
    if nargin < 4 || isempty(filterOrder)
        filterOrder = 32;
    end
    
    if nargin < 3 || isempty(mu)
        mu = 0.01;
    end
    
    N = length(noisy);
    
    % 根据噪声类型创建更好的参考信号
    if nargin < 2 || isempty(desired)
        % 尝试检测噪声特性
        Y = fft(noisy);
        P = abs(Y/N);
        P1 = P(1:floor(N/2)+1);
        freqIndex = find(P1 == max(P1));
        
        if freqIndex > 1 && freqIndex < 50 % 可能是单频干扰
            % 创建相位适配的正弦参考
            fs = 44100; % 假设采样率
            f = (freqIndex-1) * (fs/N);
            t = (0:N-1)'/fs;
            desired = sin(2*pi*f*t);
            
        else
            % 使用多重延迟方法创建参考信号
            delays = [1, 2, 4, 8, 16];
            delayedSignals = zeros(N, length(delays));
            
            for i = 1:length(delays)
                d = delays(i);
                if d < N
                    delayedSignals(d+1:end, i) = noisy(1:end-d);
                end
            end
            
            desired = mean(delayedSignals, 2);
        end
    end
    
    % 实现标准LMS算法
    x = noisy;           % 输入信号
    d = desired;         % 期望信号
    
    % 初始化权重向量和输出
    w = zeros(filterOrder, 1);  % 权重向量
    y = zeros(N, 1);     % 输出信号
    e = zeros(N, 1);     % 误差信号
    
    % LMS算法迭代实现
    for n = filterOrder:N
        % 提取输入向量
        x_n = x(n:-1:n-filterOrder+1);
        
        % 计算滤波器输出
        y(n) = w' * x_n;
        
        % 计算误差
        e(n) = d(n) - y(n);
        
        % 更新权重向量（标准LMS权重更新方程）
        w = w + mu * e(n) * x_n;
    end
    
    % 实现标准化LMS（NLMS）变种以提高收敛性能
    mu_nlms = 0.1;  % NLMS的步长因子
    w_nlms = zeros(filterOrder, 1);
    y_nlms = zeros(N, 1);
    e_nlms = zeros(N, 1);
    
    for n = filterOrder:N
        x_n = x(n:-1:n-filterOrder+1);
        
        % 计算NLMS输出
        y_nlms(n) = w_nlms' * x_n;
        
        % 计算误差
        e_nlms(n) = d(n) - y_nlms(n);
        
        % 计算归一化步长
        norm_factor = x_n' * x_n;
        if norm_factor > 0
            adaptive_mu = mu_nlms / (norm_factor + 1e-10);
        else
            adaptive_mu = mu_nlms;
        end
        
        % 更新NLMS权重
        w_nlms = w_nlms + adaptive_mu * e_nlms(n) * x_n;
    end
    
    % 结合标准LMS和NLMS的结果
    alpha = 0.7;  % 混合因子 - 偏向NLMS
    combined = alpha * e_nlms + (1-alpha) * e;
    
    % 重建滤波后的信号
    filtered = noisy - combined;
    
    % 应用后处理平滑
    b = ones(5,1)/5;
    filtered = filter(b, 1, filtered);
    
    % 归一化输出
    if max(abs(filtered)) > 0
        filtered = filtered / max(abs(filtered));
    end
end

function denoised = applyWaveletDenoising(noisy, wavelet, level)
    % 应用小波去噪
    % wavelet: 小波名称（例如，'db4', 'sym8'）
    % level: 分解层数
    
    % 执行小波分解
    [c, l] = wavedec(noisy, level, wavelet);
    
    % 估计噪声水平
    sigma = median(abs(c(l(1)+1:l(1)+l(2))))/0.6745;
    
    % 应用软阈值
    threshold = sigma * sqrt(2*log(length(noisy)));
    c_denoised = wthresh(c, 's', threshold);
    
    % 重建信号
    denoised = waverec(c_denoised, l, wavelet);
end

function filtered = applyNotchFilter(signal, fs, notchFreq, Q)
    % 应用陷波滤波器去除正弦干扰
    % fs: 采样频率
    % notchFreq: 要去除的频率(Hz)
    % Q: 品质因数（Q越高，陷波越窄）
    
    w0 = notchFreq/(fs/2);
    bw = w0/Q;
    [b, a] = iirnotch(w0, bw);
    
    filtered = filter(b, a, signal);
end