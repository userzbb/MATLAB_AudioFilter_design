% filepath: g:\zizim\Documents\code\matlab_project\demo_2\audio_processing.m
% 音频处理工具函数

function [varargout] = audio_processing(operation, varargin)
    % 音频处理主函数，作为工具函数的接口
    % operation: 操作类型，如 'addWhiteNoise', 'addNarrowbandNoise', 等
    % varargin: 根据操作类型传递给各子函数的参数
    % varargout: 根据操作类型返回的输出参数
    
    % 检查输入参数数量
    if nargin < 1
        error('输入参数不足: 至少需要提供一个操作类型参数。使用方法: audio_processing(operation, [其他参数...])');
    end
    
    % 确定要返回的输出参数数量
    num_outputs = nargout;
    
    % 根据操作类型调用相应的子函数
    switch operation
        case 'addWhiteNoise'
            % 检查此操作所需的参数数量
            if length(varargin) < 2
                error('addWhiteNoise操作需要至少2个参数: audio_processing(''addWhiteNoise'', signal, SNR_dB)');
            end
            out = cell(1, num_outputs);
            [out{:}] = addWhiteNoise(varargin{:});
        case 'addNarrowbandNoise'
            % 检查此操作所需的参数数量
            if length(varargin) < 5
                error('addNarrowbandNoise操作需要至少5个参数: audio_processing(''addNarrowbandNoise'', signal, fs, f_low, f_high, SNR_dB)');
            end
            out = cell(1, num_outputs);
            [out{:}] = addNarrowbandNoise(varargin{:});
        case 'addSinusoidalNoise'
            if length(varargin) < 4
                error('addSinusoidalNoise操作需要至少4个参数: audio_processing(''addSinusoidalNoise'', signal, fs, freq, amplitude)');
            end
            out = cell(1, num_outputs);
            [out{:}] = addSinusoidalNoise(varargin{:});
        case 'applyFIRFilter'
            % 更新参数检查，允许可选的filter order参数
            if length(varargin) < 5
                error('applyFIRFilter操作需要至少5个参数: audio_processing(''applyFIRFilter'', signal, fs, type, cutoff, window_type, [filterOrder])');
            end
            out = cell(1, num_outputs);
            [out{:}] = applyFIRFilter(varargin{:});
        case 'getFilterResponse'
            % 更新参数检查，允许可选的filter order参数
            if length(varargin) < 4
                error('getFilterResponse操作需要至少4个参数: audio_processing(''getFilterResponse'', type, cutoff, fs, window_type, [filterOrder])');
            end
            out = cell(1, max(num_outputs, 2)); % 至少需要2个输出
            [out{:}] = getFilterResponse(varargin{:});
        case 'applyLMSFilter'
            if length(varargin) < 1
                error('applyLMSFilter操作需要至少1个参数: audio_processing(''applyLMSFilter'', noisy, [desired], [mu], [filterOrder])');
            end
            out = cell(1, num_outputs);
            [out{:}] = applyLMSFilter(varargin{:});
        case 'applyWaveletDenoising'
            if length(varargin) < 3
                error('applyWaveletDenoising操作需要至少3个参数: audio_processing(''applyWaveletDenoising'', noisy, wavelet, level)');
            end
            out = cell(1, num_outputs);
            [out{:}] = applyWaveletDenoising(varargin{:});
        case 'applyNotchFilter'
            if length(varargin) < 4
                error('applyNotchFilter操作需要至少4个参数: audio_processing(''applyNotchFilter'', signal, fs, notchFreq, Q)');
            end
            out = cell(1, num_outputs);
            [out{:}] = applyNotchFilter(varargin{:});
        otherwise
            error('未知的操作类型: %s。有效的操作类型包括: ''addWhiteNoise'', ''addNarrowbandNoise'', ''addSinusoidalNoise'', ''applyFIRFilter'', ''getFilterResponse'', ''applyLMSFilter'', ''applyWaveletDenoising'', ''applyNotchFilter''', operation);
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

function filtered = applyFIRFilter(signal, fs, type, cutoff, window_type, filterOrder)
    % 使用窗函数法应用FIR滤波器
    % type: 'low'(低通), 'high'(高通), 'bandpass'(带通), 或 'stop'(带阻)
    % cutoff: 低通/高通的截止频率，或带通/带阻的[低频 高频]
    % window_type: '巴特利特窗', '汉宁窗', '汉明窗', '布莱克曼窗', 或 '凯泽窗'
    % filterOrder: 可选，滤波器阶数（默认为50，上限为150）
    
    % 滤波器阶数（带阻和带通滤波器应为偶数）
    if nargin < 6 || isempty(filterOrder)
        filterOrder = 50; % 使用较低的默认阶数
    else
        % 限制滤波器阶数范围，防止过高阶数导致显示问题
        filterOrder = min(max(filterOrder, 10), 150); 
    end
    
    % 确保带阻和带通滤波器使用偶数阶
    if (strcmp(type, 'stop') || strcmp(type, 'bandpass')) && mod(filterOrder, 2) == 1
        filterOrder = filterOrder + 1;
    end
    
    % 创建窗
    switch window_type
        case '巴特利特窗'
            win = bartlett(filterOrder+1);
        case '汉宁窗'
            win = hann(filterOrder+1);
        case '汉明窗'
            win = hamming(filterOrder+1);
        case '布莱克曼窗'
            win = blackman(filterOrder+1);
        case '凯泽窗'
            win = kaiser(filterOrder+1, 5); % Beta = 5
        otherwise
            % 如果窗函数不明确，默认使用汉明窗
            win = hamming(filterOrder+1);
    end
    
    % 设计滤波器
    nyquist = fs/2;
    switch type
        case 'low'
            cutoff_norm = cutoff/nyquist;
            b = fir1(filterOrder, cutoff_norm, 'low', win);
        case 'high'
            cutoff_norm = cutoff/nyquist;
            b = fir1(filterOrder, cutoff_norm, 'high', win);
        case 'bandpass'
            % 添加带通滤波器设计
            if ~isscalar(cutoff) && length(cutoff) == 2
                cutoff_norm = cutoff/nyquist;
                b = fir1(filterOrder, cutoff_norm, 'bandpass', win);
            else
                error('带通滤波器需要两个截止频率 [f_low, f_high]');
            end
        case 'stop'
            cutoff_norm = cutoff/nyquist;
            b = fir1(filterOrder, cutoff_norm, 'stop', win);
        otherwise
            error('不支持的滤波器类型: %s。支持的类型包括: ''low'', ''high'', ''bandpass'', ''stop''', type);
    end
    
    % 计算群延迟（大约为滤波器阶数的一半）
    groupDelay = filterOrder / 2;
    
    % 使用filtfilt替代filter以消除相位延迟（零相位滤波）
    filtered = filtfilt(b, 1, signal);
    
    % 确保输出信号有合理的幅度
    if max(abs(filtered)) < 0.01 && max(abs(signal)) > 0.01
        % 如果滤波后信号幅度太小，恢复到与原始信号相似的幅度
        scale_factor = max(abs(signal)) / max(abs(filtered));
        filtered = filtered * scale_factor * 0.8; % 稍微降低以避免削波
    end
    
    % 归一化以防止削波
    if max(abs(filtered)) > 1
        filtered = filtered / max(abs(filtered));
    end
end

function [h, w] = getFilterResponse(type, cutoff, fs, window_type, filterOrder)
    % 计算滤波器频率响应
    % 返回幅度响应h和频率w
    % filterOrder: 可选，滤波器阶数（默认为50，上限为150）
    
    if nargin < 5 || isempty(filterOrder)
        filterOrder = 50; % 使用较低的默认阶数
    else
        % 限制滤波器阶数范围
        filterOrder = min(max(filterOrder, 10), 150);
    end
    
    % 确保带阻和带通滤波器使用偶数阶
    if (strcmp(type, 'stop') || strcmp(type, 'bandpass')) && mod(filterOrder, 2) == 1
        filterOrder = filterOrder + 1;
    end
    
    % 创建窗
    switch window_type
        case '巴特利特窗'
            win = bartlett(filterOrder+1);
        case '汉宁窗'
            win = hann(filterOrder+1);
        case '汉明窗'
            win = hamming(filterOrder+1);
        case '布莱克曼窗'
            win = blackman(filterOrder+1);
        case '凯泽窗'
            win = kaiser(filterOrder+1, 5); % Beta = 5
        otherwise
            % 如果窗函数不明确，默认使用汉明窗
            win = hamming(filterOrder+1);
    end
    
    % 设计滤波器
    nyquist = fs/2;
    switch type
        case 'low'
            cutoff_norm = cutoff/nyquist;
            b = fir1(filterOrder, cutoff_norm, 'low', win);
        case 'high'
            cutoff_norm = cutoff/nyquist;
            b = fir1(filterOrder, cutoff_norm, 'high', win);
        case 'bandpass'
            % 添加带通滤波器响应计算
            if ~isscalar(cutoff) && length(cutoff) == 2
                cutoff_norm = cutoff/nyquist;
                b = fir1(filterOrder, cutoff_norm, 'bandpass', win);
            else
                error('带通滤波器需要两个截止频率 [f_low, f_high]');
            end
        case 'stop'
            cutoff_norm = cutoff/nyquist;
            b = fir1(filterOrder, cutoff_norm, 'stop', win);
        otherwise
            error('不支持的滤波器类型: %s。支持的类型包括: ''low'', ''high'', ''bandpass'', ''stop''', type);
    end
    
    % 计算频率响应
    [h, w] = freqz(b, 1, 1024);
end

function filtered = applyLMSFilter(noisy, desired, mu, filterOrder)
    % 应用自定义LMS高级滤波处理器，基于博客实现
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
            fs = 48000; % 假设采样率
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