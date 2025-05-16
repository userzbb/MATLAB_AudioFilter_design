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

function filtered = applyLMSFilter(noisy, desired, mu, filterOrder, noiseType, fs)
    % 应用LMS自适应滤波器，增强版
    % noisy: 带噪声信号
    % desired: 期望信号（若为空，则自动生成参考信号）
    % mu: 步长参数
    % filterOrder: 滤波器长度
    % noiseType: 噪声类型，可为 'white', 'narrowband', 'sinusoidal' 或 []
    % fs: 采样频率 (当noiseType不为空时必需)
    
    % 默认参数处理
    if nargin < 6 || isempty(fs)
        fs = 44100; % 默认采样率
    end
    
    if nargin < 5
        noiseType = [];
    end
    
    % 根据噪声类型优化滤波器参数
    if isempty(mu)
        % 根据噪声类型自动设置步长
        if strcmp(noiseType, 'white')
            mu = 0.005; % 白噪声需要较小步长
        elseif strcmp(noiseType, 'narrowband') 
            mu = 0.01;  % 窄带噪声可用较大步长
        elseif strcmp(noiseType, 'sinusoidal')
            mu = 0.02;  % 单频干扰可用更大步长
        else
            mu = 0.008; % 默认值
        end
    end
    
    if isempty(filterOrder)
        % 根据噪声类型自动设置滤波器阶数
        if strcmp(noiseType, 'white')
            filterOrder = 128;  % 白噪声需要较高阶
        elseif strcmp(noiseType, 'narrowband')
            filterOrder = 64;   % 窄带噪声需要中等阶数
        elseif strcmp(noiseType, 'sinusoidal')
            filterOrder = 32;   % 单频干扰可用较低阶
        else
            filterOrder = 64;   % 默认值
        end
    end
    
    % 创建更好的参考信号
    if nargin < 2 || isempty(desired)
        if strcmp(noiseType, 'sinusoidal')
            % 对于单频干扰，生成更精确的参考信号
            N = length(noisy);
            t = (0:N-1)/fs;
            sinFreq = 1500; % 默认频率，可以通过频谱分析更精确地确定
            
            % 简单的频率估计 - 实际应用中可以更复杂
            Y = fft(noisy);
            P2 = abs(Y);
            P1 = P2(1:floor(length(noisy)/2)+1);
            f = fs*(0:(length(noisy)/2))/length(noisy);
            [~, idx] = max(P1(2:end)); % 忽略直流分量
            estFreq = f(idx+1);  % 估计频率
            
            if estFreq > 100 && estFreq < fs/2-100
                sinFreq = estFreq;
            end
            
            % 创建正弦参考
            desired = sin(2*pi*sinFreq*t)';
            
        elseif strcmp(noiseType, 'narrowband')
            % 窄带噪声使用带通滤波的参考
            wn = randn(size(noisy));
            nyquist = fs/2;
            [b, a] = butter(4, [1000/nyquist, 2000/nyquist], 'bandpass');
            desired = filter(b, a, wn);
            
            % 归一化
            desired = desired / max(abs(desired));
            
        else
            % 改进的延迟线方法 - 使用多个延迟并平均
            delay_base = 5;
            num_delays = 3;
            delayed_signals = zeros(length(noisy), num_delays);
            
            for i = 1:num_delays
                d = delay_base * i;
                if d < length(noisy)
                    delayed_signals(d+1:end, i) = noisy(1:end-d);
                end
            end
            
            desired = mean(delayed_signals, 2);
        end
    end
    
    % 匹配长度
    minLen = min(length(noisy), length(desired));
    noisy = noisy(1:minLen);
    desired = desired(1:minLen);
    
    % 使用标准化LMS算法以提高性能
    nlms_filter = dsp.LMSFilter(filterOrder, 'StepSize', mu, ...
                               'Method', 'Normalized LMS', ...
                               'WeightsOutputPort', true, ...
                               'LeakageFactor', 0.9999);  % 轻微的权重泄漏以增加稳定性
    
    % 应用滤波器进行信号处理
    [filtered, weights] = step(nlms_filter, noisy, desired);
    
    % 可选：进行后处理以改善结果
    % 例如，应用一个低通滤波器去除可能引入的高频噪声
    if strcmp(noiseType, 'white')
        [b, a] = butter(4, 4000/(fs/2), 'low');
        filtered = filtfilt(b, a, filtered); % 零相位滤波
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