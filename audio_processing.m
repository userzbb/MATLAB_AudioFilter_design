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
    % 应用LMS自适应滤波器
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
    
    if nargin < 2 || isempty(desired)
        % 使用延迟的噪声信号作为参考
        delay = 5;
        N = length(noisy);
        desired = zeros(size(noisy));
        if delay < N
            desired(delay+1:end) = noisy(1:end-delay);
        end
    end
    
    % 匹配长度
    minLen = min(length(noisy), length(desired));
    noisy = noisy(1:minLen);
    desired = desired(1:minLen);
    
    % 创建LMS滤波器
    lms_filter = dsp.LMSFilter(filterOrder, 'StepSize', mu);
    
    % 应用滤波器
    filtered = step(lms_filter, noisy, desired);
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