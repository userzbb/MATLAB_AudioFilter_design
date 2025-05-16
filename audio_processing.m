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
            % 为了兼容性，将applyLMSFilter重定向到applyAdaptiveFilter
            out = cell(1, num_outputs);
            [out{:}] = applyAdaptiveFilter(varargin{:});
        case 'applyAdaptiveFilter'
            out = cell(1, num_outputs);
            [out{:}] = applyAdaptiveFilter(varargin{:});
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

function filtered = applyAdaptiveFilter(noisy, desired, step_size, filter_length, algorithm_type)
    % 高级自适应滤波算法实现
    % noisy: 带噪声信号
    % desired: 期望信号（若为空，则自动生成参考信号）
    % step_size: 步长参数（若为空，则自动优化）
    % filter_length: 滤波器长度（若为空，则自动选择）
    % algorithm_type: 算法类型（若为空，则自动选择最优算法）
    
    % 参数预处理
    if nargin < 5 || isempty(algorithm_type)
        algorithm_type = 'auto'; % 自动选择最佳算法
    end
    
    if nargin < 4 || isempty(filter_length)
        filter_length = 64; % 默认滤波器长度
    end
    
    if nargin < 3 || isempty(step_size)
        step_size = []; % 自动选择步长
    end
    
    % 确保信号为列向量
    noisy = noisy(:);
    fs = 44100; % 默认采样率，实际应用中应根据情况设置
    
    % 创建参考信号（如果未提供）
    if nargin < 2 || isempty(desired)
        [noiseType, refSignal] = createReferenceSignal(noisy, fs);
    else
        refSignal = desired(:);
        % 进行简单分析以确定噪声类型
        noiseType = analyzeNoiseType(noisy, fs);
    end
    
    % 基于噪声类型自动选择算法
    if strcmpi(algorithm_type, 'auto')
        algorithm_type = selectOptimalAlgorithm(noiseType);
    end
    
    % 根据算法类型和噪声特性选择最佳参数
    [optFilter_length, optStep_size, forgettingFactor] = ...
        selectOptimalParameters(algorithm_type, noiseType, noisy, refSignal, filter_length, step_size);
    
    % 预处理信号以提高滤波效果
    [processedNoisy, processedRef] = preprocessSignals(noisy, refSignal, noiseType, fs);
    
    % 应用选择的自适应滤波算法
    switch lower(algorithm_type)
        case 'lms'
            [filtered, errors, weights] = applyLMS(processedNoisy, processedRef, optStep_size, optFilter_length);
        case 'nlms'
            [filtered, errors, weights] = applyNLMS(processedNoisy, processedRef, optStep_size, optFilter_length);
        case 'rls'
            [filtered, errors, weights] = applyRLS(processedNoisy, processedRef, forgettingFactor, optFilter_length);
        case 'vsslms'
            [filtered, errors, weights] = applyVSSLMS(processedNoisy, processedRef, optStep_size, optFilter_length);
        otherwise
            % 默认使用NLMS
            [filtered, errors, weights] = applyNLMS(processedNoisy, processedRef, optStep_size, optFilter_length);
    end
    
    % 后处理以进一步提高信号质量
    filtered = postprocessSignal(filtered, noisy, noiseType, fs);
    
    % 归一化输出
    if max(abs(filtered)) > 0
        filtered = filtered / max(abs(filtered));
    end
    
    % 评估性能（调试用）
    % evaluatePerformance(noisy, refSignal, filtered, errors, weights, algorithm_type);
end

function [noiseType, refSignal] = createReferenceSignal(noisy, fs)
    % 分析噪声类型并创建相应的参考信号
    noiseType = analyzeNoiseType(noisy, fs);
    N = length(noisy);
    
    switch noiseType
        case 'sinusoidal'
            % 检测主频率
            [freq, phase] = detectSinusoidalParameters(noisy, fs);
            t = (0:N-1)'/fs;
            refSignal = sin(2*pi*freq*t + phase);
            
        case 'narrowband'
            % 检测主频带
            [centerFreq, bandwidth] = detectNarrowbandParameters(noisy, fs);
            % 生成带通滤波的白噪声参考
            rawNoise = randn(N, 1);
            [b, a] = butter(4, [max(10, centerFreq-bandwidth/2)/(fs/2), ...
                              min(fs/2-10, centerFreq+bandwidth/2)/(fs/2)], 'bandpass');
            refSignal = filtfilt(b, a, rawNoise);
            
        case 'white'
            % 使用延迟线方法
            delays = [1, 2, 4, 8, 16, 32];
            delayedSignals = zeros(N, length(delays));
            
            for i = 1:length(delays)
                d = delays(i);
                if d < N
                    delayedSignals(d+1:end, i) = noisy(1:end-d);
                end
            end
            
            refSignal = mean(delayedSignals, 2);
            
            % 预白化处理
            [b, a] = butter(4, 0.9, 'high');
            refSignal = filtfilt(b, a, refSignal);
            
        otherwise
            % 默认为延迟线参考
            refSignal = zeros(size(noisy));
            delay = 1;
            if N > delay
                refSignal(delay+1:end) = noisy(1:end-delay);
            end
    end
    
    % 归一化参考信号
    refSignal = refSignal / sqrt(mean(refSignal.^2));
end

function noiseType = analyzeNoiseType(signal, fs)
    % 分析信号以确定主要噪声类型
    N = length(signal);
    
    % 计算频谱
    NFFT = 2^nextpow2(N);
    Y = fft(signal, NFFT);
    P2 = abs(Y/N);
    P1 = P2(1:NFFT/2+1);
    f = fs * (0:(NFFT/2))/NFFT;
    
    % 寻找主要峰值
    [pks, locs] = findpeaks(P1, 'MinPeakHeight', 0.5*max(P1), 'SortStr', 'descend');
    
    % 判断噪声类型
    if ~isempty(pks) && length(pks) >= 1
        mainPeakIdx = locs(1);
        mainPeakFreq = f(mainPeakIdx);
        mainPeakAmp = pks(1);
        
        % 计算峰值能量比
        totalEnergy = sum(P1.^2);
        peakEnergy = mainPeakAmp^2;
        peakEnergyRatio = peakEnergy / totalEnergy;
        
        % 单频干扰: 明显的单一峰值，能量集中
        if peakEnergyRatio > 0.4 && mainPeakFreq > 100  % 排除低频分量
            noiseType = 'sinusoidal';
            return;
        end
        
        % 窄带噪声: 在特定频段有多个峰值
        narrowbandPeaks = sum(f(locs) >= 1000 & f(locs) <= 2000);
        if narrowbandPeaks >= 2
            noiseType = 'narrowband';
            return;
        end
    end
    
    % 默认为白噪声
    noiseType = 'white';
end

function [freq, phase] = detectSinusoidalParameters(signal, fs)
    % 检测正弦信号的频率和相位
    N = length(signal);
    
    % 获取频谱
    NFFT = 2^nextpow2(N);
    Y = fft(signal, NFFT);
    P2 = abs(Y/N);
    P1 = P2(1:NFFT/2+1);
    f = fs * (0:(NFFT/2))/NFFT;
    
    % 查找主频率
    [~, maxIdx] = max(P1(2:end)); % 跳过直流分量
    freq = f(maxIdx+1);
    
    % 估计相位
    phases = 0:pi/12:2*pi-pi/12;
    t = (0:N-1)'/fs;
    maxCorr = -inf;
    bestPhase = 0;
    
    for p = phases
        testSig = sin(2*pi*freq*t + p);
        corr = abs(sum(signal .* testSig));
        if corr > maxCorr
            maxCorr = corr;
            bestPhase = p;
        end
    end
    
    phase = bestPhase;
end

function [centerFreq, bandwidth] = detectNarrowbandParameters(signal, fs)
    % 检测窄带噪声的中心频率和带宽
    N = length(signal);
    
    % 获取频谱
    NFFT = 2^nextpow2(N);
    Y = fft(signal, NFFT);
    P2 = abs(Y/N);
    P1 = P2(1:NFFT/2+1);
    f = fs * (0:(NFFT/2))/NFFT;
    
    % 查找窄带内的峰值
    [pks, locs] = findpeaks(P1, 'MinPeakHeight', 0.3*max(P1));
    
    % 筛选1000-2000Hz范围内的峰值
    narrowbandIdx = f(locs) >= 1000 & f(locs) <= 2000;
    narrowbandFreqs = f(locs(narrowbandIdx));
    
    if ~isempty(narrowbandFreqs)
        centerFreq = mean(narrowbandFreqs);
        
        % 估计带宽
        if length(narrowbandFreqs) > 1
            bandwidth = max(narrowbandFreqs) - min(narrowbandFreqs) + 200; % 额外200Hz余量
        else
            bandwidth = 500; % 默认500Hz带宽
        end
    else
        % 默认值
        centerFreq = 1500;
        bandwidth = 1000;
    end
end

function algorithm = selectOptimalAlgorithm(noiseType)
    % 根据噪声类型选择最佳算法
    switch noiseType
        case 'sinusoidal'
            algorithm = 'nlms'; % 单频干扰简单且收敛快
        case 'narrowband'
            algorithm = 'rls';  % 窄带噪声复杂性高，RLS效果好
        case 'white'
            algorithm = 'vsslms'; % 白噪声环境变化大，变步长效果好
        otherwise
            algorithm = 'nlms'; % 默认使用NLMS
    end
end

function [optLength, optStep, forgettingFactor] = selectOptimalParameters(algorithm, noiseType, signal, refSignal, length, step)
    % 根据算法类型和噪声特性选择最佳参数
    
    % 初始化参数
    optLength = length;
    optStep = step;
    forgettingFactor = 0.999; % RLS算法默认遗忘因子
    
    % 分析信号功率
    signalPower = mean(signal.^2);
    
    switch lower(algorithm)
        case 'lms'
            % LMS参数优化
            switch noiseType
                case 'sinusoidal'
                    optLength = min(32, length); % 单频需要较少滤波器阶数
                    if isempty(step)
                        optStep = 0.05; % 单频可以用较大步长
                    end
                case 'narrowband'
                    optLength = min(64, max(48, length));
                    if isempty(step)
                        optStep = 0.01; % 窄带噪声步长适中
                    end
                case 'white'
                    optLength = min(128, max(64, length));
                    if isempty(step)
                        optStep = 0.005; % 白噪声需要较小步长
                    end
            end
            
        case 'nlms'
            % NLMS参数优化
            switch noiseType
                case 'sinusoidal'
                    optLength = min(32, length);
                    if isempty(step)
                        optStep = 0.5; % NLMS可用更大步长
                    end
                case 'narrowband'
                    optLength = min(64, max(48, length));
                    if isempty(step)
                        optStep = 0.2;
                    end
                case 'white'
                    optLength = min(128, max(64, length));
                    if isempty(step)
                        optStep = 0.1;
                    end
            end
            
        case 'rls'
            % RLS参数优化
            switch noiseType
                case 'sinusoidal'
                    optLength = min(16, length); % RLS计算复杂度高，需较小阶数
                    forgettingFactor = 0.99; % 快速收敛
                case 'narrowband'
                    optLength = min(32, length);
                    forgettingFactor = 0.995;
                case 'white'
                    optLength = min(64, length);
                    forgettingFactor = 0.998; % 接近1的遗忘因子
            end
            
        case 'vsslms'
            % 变步长LMS参数优化
            switch noiseType
                case 'sinusoidal'
                    optLength = min(32, length);
                    if isempty(step)
                        optStep = [0.1, 0.001]; % [最大步长, 最小步长]
                    elseif isscalar(step)
                        % 如果输入是标量，转换为范围
                        optStep = [step*10, step/10];
                    end
                case 'narrowband'
                    optLength = min(64, max(48, length));
                    if isempty(step)
                        optStep = [0.05, 0.0005];
                    elseif isscalar(step)
                        optStep = [step*5, step/20];
                    end
                case 'white'
                    optLength = min(128, max(64, length));
                    if isempty(step)
                        optStep = [0.02, 0.0001];
                    elseif isscalar(step)
                        optStep = [step*2, step/100];
                    end
            end
            
            % 确保VSSLMS的步长为二元组
            if ~isempty(optStep) && ~isnumeric(optStep)
                optStep = [0.05, 0.0005]; % 默认值
            elseif isscalar(optStep)
                optStep = [optStep*5, optStep/20]; % 将单一值转换为范围
            end
    end
end

function [processedNoisy, processedRef] = preprocessSignals(noisy, refSignal, noiseType, fs)
    % 预处理信号以提高滤波效果
    
    % 默认不做处理
    processedNoisy = noisy;
    processedRef = refSignal;
    
    switch noiseType
        case 'sinusoidal'
            % 单频干扰不需特殊预处理
            
        case 'narrowband'
            % 对窄带噪声应用温和的滤波
            [b, a] = butter(2, [500/(fs/2), 3000/(fs/2)], 'bandpass');
            processedNoisy = filtfilt(b, a, noisy);
            processedRef = filtfilt(b, a, refSignal);
            
        case 'white'
            % 白噪声去除直流偏置并应用高通滤波以改善收敛
            processedNoisy = noisy - mean(noisy);
            processedRef = refSignal - mean(refSignal);
            
            % 高通滤波去除低频漂移
            [b, a] = butter(2, 100/(fs/2), 'high');
            processedNoisy = filtfilt(b, a, processedNoisy);
            processedRef = filtfilt(b, a, processedRef);
    end
end

function [filtered, errors, weights] = applyLMS(x, d, mu, M)
    % 标准LMS算法实现
    N = length(x);
    
    % 初始化
    w = zeros(M, 1);     % 初始权重向量
    y = zeros(N, 1);     % 滤波器输出
    e = zeros(N, 1);     % 误差信号
    W = zeros(M, N);     % 所有权重历史
    
    % LMS算法主循环
    for n = M:N
        % 获取当前输入向量
        x_n = x(n:-1:n-M+1);
        
        % 计算输出
        y(n) = w' * x_n;
        
        % 计算误差
        e(n) = d(n) - y(n);
        
        % 更新权重
        w = w + mu * e(n) * x_n;
        
        % 存储权重历史
        W(:, n) = w;
    end
    
    % 使用最终收敛权重重新滤波
    filtered = zeros(N, 1);
    final_w = w;
    
    for n = M:N
        x_n = x(n:-1:n-M+1);
        filtered(n) = final_w' * x_n;
    end
    
    % 初始M-1个样本直接使用输入
    filtered(1:M-1) = x(1:M-1);
    
    % 返回误差和权重历史
    errors = e;
    weights = W;
end

function [filtered, errors, weights] = applyNLMS(x, d, mu, M)
    % 规范化LMS (NLMS) 算法实现
    N = length(x);
    
    % 初始化
    w = zeros(M, 1);     % 初始权重向量
    y = zeros(N, 1);     % 滤波器输出
    e = zeros(N, 1);     % 误差信号
    W = zeros(M, N);     % 所有权重历史
    
    % NLMS算法主循环
    for n = M:N
        % 获取当前输入向量
        x_n = x(n:-1:n-M+1);
        
        % 计算输出
        y(n) = w' * x_n;
        
        % 计算误差
        e(n) = d(n) - y(n);
        
        % 计算归一化步长
        power = x_n' * x_n;
        if power > 0
            adaptive_mu = mu / (power + 1e-10);
        else
            adaptive_mu = mu;
        end
        
        % 更新权重
        w = w + adaptive_mu * e(n) * x_n;
        
        % 存储权重历史
        W(:, n) = w;
    end
    
    % 使用最终收敛权重重新滤波
    filtered = zeros(N, 1);
    final_w = w;
    
    for n = M:N
        x_n = x(n:-1:n-M+1);
        filtered(n) = final_w' * x_n;
    end
    
    % 初始M-1个样本直接使用输入
    filtered(1:M-1) = x(1:M-1);
    
    % 返回误差和权重历史
    errors = e;
    weights = W;
end

function [filtered, errors, weights] = applyRLS(x, d, lambda, M)
    % 递归最小二乘 (RLS) 算法实现
    N = length(x);
    
    % 初始化
    w = zeros(M, 1);         % 初始权重向量
    y = zeros(N, 1);         % 滤波器输出
    e = zeros(N, 1);         % 误差信号
    W = zeros(M, N);         % 所有权重历史
    
    % 初始化逆相关矩阵
    delta = 0.001;           % 正则化参数
    P = eye(M) / delta;      % 逆相关矩阵
    
    % RLS算法主循环
    for n = M:N
        % 获取当前输入向量
        x_n = x(n:-1:n-M+1);
        
        % 计算中间向量 (Kalman增益)
        k = P * x_n / (lambda + x_n' * P * x_n);
        
        % 计算输出
        y(n) = w' * x_n;
        
        % 计算误差
        e(n) = d(n) - y(n);
        
        % 更新权重
        w = w + k * e(n);
        
        % 更新逆相关矩阵
        P = (P - k * x_n' * P) / lambda;
        
        % 存储权重历史
        W(:, n) = w;
    end
    
    % 使用最终收敛权重重新滤波
    filtered = zeros(N, 1);
    final_w = w;
    
    for n = M:N
        x_n = x(n:-1:n-M+1);
        filtered(n) = final_w' * x_n;
    end
    
    % 初始M-1个样本直接使用输入
    filtered(1:M-1) = x(1:M-1);
    
    % 返回误差和权重历史
    errors = e;
    weights = W;
end

function [filtered, errors, weights] = applyVSSLMS(x, d, mu_range, M)
    % 变步长LMS (VSS-LMS) 算法实现
    % mu_range: [mu_max, mu_min] - 最大和最小步长
    
    N = length(x);
    
    % 确保mu_range是二元组，处理标量输入
    if isscalar(mu_range)
        mu_max = mu_range * 5;  % 默认最大步长为输入的5倍
        mu_min = mu_range / 20; % 默认最小步长为输入的1/20
    else
        % 提取步长范围
        mu_max = mu_range(1);
        mu_min = mu_range(2);
    end
    
    % 初始化
    w = zeros(M, 1);     % 初始权重向量
    y = zeros(N, 1);     % 滤波器输出
    e = zeros(N, 1);     % 误差信号
    W = zeros(M, N);     % 所有权重历史
    
    % 初始步长设置为最大值
    mu = mu_max;
    
    % 误差平方窗口大小和控制参数
    window_size = 10;
    error_window = zeros(window_size, 1);
    alpha = 0.97;        % 步长调整因子
    
    % VSS-LMS算法主循环
    for n = M:N
        % 获取当前输入向量
        x_n = x(n:-1:n-M+1);
        
        % 计算输出
        y(n) = w' * x_n;
        
        % 计算误差
        e(n) = d(n) - y(n);
        
        % 更新误差窗口
        error_window = [e(n)^2; error_window(1:end-1)];
        
        % 计算当前和上一窗口的误差平方和
        if n > M + window_size
            % 根据误差变化调整步长
            curr_error = mean(error_window);
            
            % 如果误差增加，减小步长；如果误差减小，增加步长
            if n > M + 2*window_size
                prev_error = mean(e(n-window_size:n-1).^2);
                
                if curr_error > prev_error
                    mu = max(mu * alpha, mu_min);  % 误差增加，减小步长
                else
                    mu = min(mu / alpha, mu_max);  % 误差减小，增加步长
                end
            end
        end
        
        % 更新权重
        w = w + mu * e(n) * x_n;
        
        % 存储权重历史
        W(:, n) = w;
    end
    
    % 使用最终收敛权重重新滤波
    filtered = zeros(N, 1);
    final_w = w;
    
    for n = M:N
        x_n = x(n:-1:n-M+1);
        filtered(n) = final_w' * x_n;
    end
    
    % 初始M-1个样本直接使用输入
    filtered(1:M-1) = x(1:M-1);
    
    % 返回误差和权重历史
    errors = e;
    weights = W;
end

function output = postprocessSignal(filtered, original, noiseType, fs)
    % 后处理增强滤波效果
    
    % 默认不做处理
    output = filtered;
    
    switch noiseType
        case 'sinusoidal'
            % 陷波滤波器补充单频干扰去除
            [freq, ~] = detectSinusoidalParameters(original, fs);
            [b, a] = iirnotch(freq/(fs/2), freq/(fs/2)/35);
            output = filtfilt(b, a, filtered);
            
        case 'narrowband'
            % 窄带噪声可能需要额外带阻滤波器
            [centerFreq, bandwidth] = detectNarrowbandParameters(original, fs);
            [b, a] = butter(4, [max(50, centerFreq-bandwidth/2)/(fs/2), ...
                              min(fs/2-50, centerFreq+bandwidth/2)/(fs/2)], 'stop');
            output = filtfilt(b, a, filtered);
            
            % 平滑过滤以减少伪影
            window = hamming(5)/sum(hamming(5));
            output = filter(window, 1, output);
            
        case 'white'
            % 低通滤波去除高频残留噪声
            [b, a] = butter(4, 4000/(fs/2), 'low');
            output = filtfilt(b, a, filtered);
            
            % 混合部分原始信号可提高语音自然度
            alpha = 0.85;
            output = alpha * output + (1-alpha) * original;
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