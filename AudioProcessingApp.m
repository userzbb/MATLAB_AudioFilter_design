% filepath: g:\zizim\Documents\code\matlab_project\demo_2\AudioProcessingApp.m
classdef AudioProcessingApp < matlab.apps.AppBase

    % 对应于应用组件的属性
    properties (Access = public)
        UIFigure           matlab.ui.Figure
        TabGroup           matlab.ui.container.TabGroup
        OriginalTab        matlab.ui.container.Tab
        NoiseTab           matlab.ui.container.Tab
        FilteringTab       matlab.ui.container.Tab
        IIRTab             matlab.ui.container.Tab
        AdaptiveTab        matlab.ui.container.Tab
        
        % 原始音频面板
        OriginalPanel      matlab.ui.container.Panel
        LoadButton         matlab.ui.control.Button
        PlayOrigButton     matlab.ui.control.Button
        StopOrigButton     matlab.ui.control.Button
        AudioInfoLabel     matlab.ui.control.Label
        TimeAxes           matlab.ui.control.UIAxes
        FreqAxes           matlab.ui.control.UIAxes
        
        % 噪声面板
        NoisePanel         matlab.ui.container.Panel
        AddNoiseButton     matlab.ui.control.Button
        NoiseTypeDropDown  matlab.ui.control.DropDown
        PlayNoisyButton    matlab.ui.control.Button
        StopNoisyButton    matlab.ui.control.Button
        NoiseTimeAxes      matlab.ui.control.UIAxes
        NoiseFreqAxes      matlab.ui.control.UIAxes
        
        % FIR滤波器面板
        FilterPanel        matlab.ui.container.Panel
        FilterTypeDropDown matlab.ui.control.DropDown
        WindowTypeDropDown matlab.ui.control.DropDown
        FilterButton       matlab.ui.control.Button
        FilterResponseAxes matlab.ui.control.UIAxes
        FilteredTimeAxes   matlab.ui.control.UIAxes
        FilteredFreqAxes   matlab.ui.control.UIAxes
        PlayFilteredButton matlab.ui.control.Button
        StopFilteredButton matlab.ui.control.Button
        
        % IIR滤波器面板
        IIRPanel           matlab.ui.container.Panel
        IIRFilterTypeDropDown matlab.ui.control.DropDown
        IIRDesignDropDown  matlab.ui.control.DropDown
        IIROrderEdit       matlab.ui.control.NumericEditField
        IIRCutoffEdit      matlab.ui.control.EditField
        IIRRippleEdit      matlab.ui.control.NumericEditField
        IIRAttenuationEdit matlab.ui.control.NumericEditField
        IIRNoiseTypeDropDown matlab.ui.control.DropDown
        IIRFilterButton    matlab.ui.control.Button
        IIRResponseAxes    matlab.ui.control.UIAxes
        IIRTimeAxes        matlab.ui.control.UIAxes
        IIRFreqAxes        matlab.ui.control.UIAxes
        PlayIIRButton      matlab.ui.control.Button
        StopIIRButton      matlab.ui.control.Button
        
        % 高级滤波处理面板
        AdaptivePanel      matlab.ui.container.Panel
        AlgorithmDropDown  matlab.ui.control.DropDown
        AdaptiveButton     matlab.ui.control.Button
        AdaptiveTimeAxes   matlab.ui.control.UIAxes
        AdaptiveFreqAxes   matlab.ui.control.UIAxes
        PlayAdaptiveButton matlab.ui.control.Button
        StopAdaptiveButton matlab.ui.control.Button
    end
    
    % 音频数据属性
    properties (Access = private)
        originalAudio      % 原始音频数据
        fs                 % 采样频率
        noisyAudio         % 带噪音频数据
        currentNoiseType   % 当前噪声类型
        filteredAudio      % 滤波后的音频数据
        iirFilteredAudio   % IIR滤波后的音频数据
        adaptiveAudio      % 高级滤波处理后的音频
    end
    
    methods (Access = private)
        
        % 加载音频文件的方法
        function loadAudio(app)
            [filename, pathname] = uigetfile({'*.wav;*.mp3', '音频文件 (*.wav, *.mp3)'}, '选择一个音频文件');
            if isequal(filename, 0) || isequal(pathname, 0)
                return;
            end
            
            fullpath = fullfile(pathname, filename);
            [app.originalAudio, app.fs] = audioread(fullpath);
            
            % 确保音频是单声道
            if size(app.originalAudio, 2) > 1
                app.originalAudio = mean(app.originalAudio, 2); % 转换为单声道
            end
            
            % 不再限制音频长度为10秒
            
            % 更新信息标签
            duration = length(app.originalAudio) / app.fs;
            app.AudioInfoLabel.Text = sprintf('采样率: %d Hz, 时长: %.2f 秒, 采样点数: %d', ...
                app.fs, duration, length(app.originalAudio));
            
            % 绘制原始音频
            plotAudio(app, app.originalAudio, app.TimeAxes, app.FreqAxes);
        end
        
        % 在时域和频域绘制音频的方法
        function plotAudio(app, audioData, timeAxes, freqAxes)
            % 时域图
            t = (0:length(audioData)-1) / app.fs;
            plot(timeAxes, t, audioData);
            timeAxes.XLabel.String = '时间 (秒)';
            timeAxes.YLabel.String = '幅度';
            timeAxes.Title.String = '时域波形';
            
            % 频域图
            N = length(audioData);
            Y = fft(audioData);
            P2 = abs(Y/N);
            P1 = P2(1:floor(N/2)+1);
            P1(2:end-1) = 2*P1(2:end-1);
            f = app.fs * (0:(N/2))/N;
            plot(freqAxes, f, P1);
            freqAxes.XLabel.String = '频率 (Hz)';
            freqAxes.YLabel.String = '幅度';
            freqAxes.Title.String = '频谱图';
        end
        
        % 向音频添加噪声的方法
        function addNoise(app)
            if isempty(app.originalAudio)
                uialert(app.UIFigure, '请先加载音频文件。', '无音频文件');
                return;
            end
            
            noiseType = app.NoiseTypeDropDown.Value;
            app.currentNoiseType = noiseType;
            
            switch noiseType
                case '高斯白噪声'
                    app.noisyAudio = audio_processing('addWhiteNoise', app.originalAudio, 10); % SNR = 10dB
                case '窄带噪声(1000-2000Hz)'
                    app.noisyAudio = audio_processing('addNarrowbandNoise', app.originalAudio, app.fs, 1000, 2000, 10);
                case '单频干扰(1500Hz)'
                    app.noisyAudio = audio_processing('addSinusoidalNoise', app.originalAudio, app.fs, 1500, 0.1);
            end
            
            % 绘制带噪音频
            plotAudio(app, app.noisyAudio, app.NoiseTimeAxes, app.NoiseFreqAxes);
        end
        
        % 应用滤波器的方法
        function applyFilter(app)
            if isempty(app.noisyAudio)
                uialert(app.UIFigure, '请先向音频添加噪声。', '无噪声音频');
                return;
            end
            
            filterType = app.FilterTypeDropDown.Value;
            windowType = app.WindowTypeDropDown.Value;
            
            % 根据噪声类型选择合适的滤波器阶数
            if strcmp(app.currentNoiseType, '高斯白噪声')
                filterOrder = 50;  % 白噪声需要中等阶数
            elseif strcmp(app.currentNoiseType, '窄带噪声(1000-2000Hz)')
                filterOrder = 80;  % 窄带噪声需要较高阶数以获得更陡峭的过渡带
            elseif strcmp(app.currentNoiseType, '单频干扰(1500Hz)')
                filterOrder = 40;  % 单频干扰较简单，较低阶数即可
            else
                filterOrder = 50;  % 默认中等阶数
            end
            
            % 检查是否为专用滤波器，使用专用窗函数
            if contains(filterType, '专用滤波器')
                switch filterType
                    case '白噪声专用滤波器'
                        windowType = '汉明窗';
                    case '窄带噪声专用滤波器'
                        windowType = '布莱克曼窗';
                        filterOrder = 100;  % 窄带噪声专用滤波器使用较高阶数
                    case '单频干扰专用滤波器'
                        % 陷波滤波器不需要窗函数
                        windowType = '';
                end
            end
            
            switch filterType
                case '低通滤波'
                    % 根据噪声类型确定截止频率
                    if strcmp(app.currentNoiseType, '高斯白噪声')
                        cutoff = 4000; % 白噪声的示例截止频率
                    elseif strcmp(app.currentNoiseType, '窄带噪声(1000-2000Hz)')
                        cutoff = 900; % 低于噪声频带
                    else % 单频干扰
                        cutoff = 1400; % 低于1500Hz干扰
                    end
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'low', cutoff, windowType, filterOrder);
                    
                case '高通滤波'
                    % 作为演示，使用高通滤波器
                    cutoff = 2100; % 高于噪声频率
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'high', cutoff, windowType, filterOrder);
                    
                case '带阻滤波'
                    if strcmp(app.currentNoiseType, '窄带噪声(1000-2000Hz)')
                        f1 = 950; f2 = 2050; % 略宽于噪声频带
                    else % 单频干扰
                        f1 = 1450; f2 = 1550; % 围绕1500Hz
                    end
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'stop', [f1 f2], windowType, filterOrder);
                    
                case '白噪声专用滤波器'
                    % 为白噪声设计的优化滤波器 - 使用低通滤波
                    % 语音信号大部分能量集中在4kHz以下
                    cutoff = 3500; % 略低于标准低通以提供更好的噪声抑制
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'low', cutoff, '汉明窗', 60);
                    
                case '窄带噪声专用滤波器'
                    % 为1000-2000Hz窄带噪声设计的专用滤波器
                    % 使用精确调整的带阻滤波器
                    f1 = 980; f2 = 2020; % 精确匹配噪声带宽
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'stop', [f1 f2], '布莱克曼窗', 100);
                    
                case '单频干扰专用滤波器'
                    % 为1500Hz单频干扰设计的专用滤波器
                    % 使用陷波滤波器，最适合单频干扰
                    app.filteredAudio = audio_processing('applyNotchFilter', app.noisyAudio, app.fs, 1500, 35);
            end
            
            % 绘制滤波后的音频
            plotAudio(app, app.filteredAudio, app.FilteredTimeAxes, app.FilteredFreqAxes);
            
            % 绘制滤波器频率响应
            plotFilterResponse(app, filterType, windowType, filterOrder);
        end
        
        % 绘制滤波器频率响应的方法
        function plotFilterResponse(app, filterType, windowType, filterOrder)
            % 根据当前设置创建滤波器
            switch filterType
                case '低通滤波'
                    cutoff = 4000; % 示例值，应与applyFilter中使用的值匹配
                    [h, w] = audio_processing('getFilterResponse', 'low', cutoff, app.fs, windowType, filterOrder);
                    titleStr = '低通滤波器频率响应';
                    
                case '高通滤波'
                    cutoff = 2100;
                    [h, w] = audio_processing('getFilterResponse', 'high', cutoff, app.fs, windowType, filterOrder);
                    titleStr = '高通滤波器频率响应';
                    
                case '带阻滤波'
                    f1 = 950; f2 = 2050; % 示例，应与applyFilter匹配
                    [h, w] = audio_processing('getFilterResponse', 'stop', [f1 f2], app.fs, windowType, filterOrder);
                    titleStr = '带阻滤波器频率响应';
                    
                case '白噪声专用滤波器'
                    cutoff = 3500;
                    [h, w] = audio_processing('getFilterResponse', 'low', cutoff, app.fs, '汉明窗', 60);
                    titleStr = '白噪声专用滤波器频率响应';
                    
                case '窄带噪声专用滤波器'
                    f1 = 980; f2 = 2020;
                    [h, w] = audio_processing('getFilterResponse', 'stop', [f1 f2], app.fs, '布莱克曼窗', 100);
                    titleStr = '窄带噪声专用滤波器频率响应';
                    
                case '单频干扰专用滤波器'
                    % 为了显示陷波滤波器的频率响应，我们需要手动计算
                    f = linspace(0, app.fs/2, 1024);
                    w = 2*pi*f/app.fs;
                    w0 = 2*pi*1500/app.fs;
                    bw = w0/35;
                    [b, a] = iirnotch(w0/(app.fs/2), bw/(app.fs/2));
                    [h, w] = freqz(b, a, 1024);
                    titleStr = '单频干扰专用滤波器频率响应';
            end
            
            % 绘制滤波器响应
            plot(app.FilterResponseAxes, w*app.fs/(2*pi), 20*log10(abs(h)));
            app.FilterResponseAxes.XLabel.String = '频率 (Hz)';
            app.FilterResponseAxes.YLabel.String = '幅度 (dB)';
            app.FilterResponseAxes.Title.String = titleStr;
            
            % 如果是陷波或带阻滤波器，调整y轴范围以更好地显示阻带
            if contains(filterType, '带阻') || contains(filterType, '单频干扰')
                app.FilterResponseAxes.YLim = [-80, 5];
            end
        end
        
        % 应用IIR滤波器的方法
        function applyIIRFilter(app)
            if isempty(app.noisyAudio)
                uialert(app.UIFigure, '请先向音频添加噪声。', '无噪声音频');
                return;
            end
            
            filterType = app.IIRFilterTypeDropDown.Value;
            designType = app.IIRDesignDropDown.Value;
            
            % 限制滤波器阶数范围，防止过高阶数导致显示问题
            filterOrder = app.IIROrderEdit.Value;
            % IIR滤波器的最大阶数通常应小于FIR滤波器
            filterOrder = min(max(filterOrder, 2), 20);
            app.IIROrderEdit.Value = filterOrder; % 更新UI显示
            
            % 获取噪声类型预设滤波器
            if strcmp(app.IIRNoiseTypeDropDown.Value, '自定义')
                % 从UI控件获取参数
                cutoffStr = app.IIRCutoffEdit.Value;
                passRipple = app.IIRRippleEdit.Value;
                stopAtten = app.IIRAttenuationEdit.Value;
                
                % 解析截止频率字符串
                if contains(cutoffStr, ',') || contains(cutoffStr, ' ')
                    % 带通或带阻滤波器有两个截止频率
                    cutoffParts = strsplit(strrep(cutoffStr, ',', ' '));
                    cutoffFreq = [str2double(cutoffParts{1}), str2double(cutoffParts{2})];
                else
                    % 低通或高通滤波器只有一个截止频率
                    cutoffFreq = str2double(cutoffStr);
                end
            else
                % 预设噪声特定滤波器
                [designType, cutoffFreq, filterOrder, passRipple, stopAtten] = getNoiseSpecificIIRParams(app, app.IIRNoiseTypeDropDown.Value);
                % 确保阶数也在合理范围内
                filterOrder = min(max(filterOrder, 2), 20);
            end
            
            % 归一化截止频率(0到1之间，1对应Nyquist频率fs/2)
            Wn = cutoffFreq / (app.fs/2);
            
            % 设计IIR滤波器
            try
                switch filterType
                    case '巴特沃斯'
                        switch designType
                            case '低通'
                                [b, a] = butter(filterOrder, Wn, 'low');
                            case '高通'
                                [b, a] = butter(filterOrder, Wn, 'high');
                            case '带通'
                                [b, a] = butter(filterOrder, Wn, 'bandpass');
                            case '带阻'
                                [b, a] = butter(filterOrder, Wn, 'stop');
                        end
                        
                    case '切比雪夫I型'
                        switch designType
                            case '低通'
                                [b, a] = cheby1(filterOrder, passRipple, Wn, 'low');
                            case '高通'
                                [b, a] = cheby1(filterOrder, passRipple, Wn, 'high');
                            case '带通'
                                [b, a] = cheby1(filterOrder, passRipple, Wn, 'bandpass');
                            case '带阻'
                                [b, a] = cheby1(filterOrder, passRipple, Wn, 'stop');
                        end
                        
                    case '切比雪夫II型'
                        switch designType
                            case '低通'
                                [b, a] = cheby2(filterOrder, stopAtten, Wn, 'low');
                            case '高通'
                                [b, a] = cheby2(filterOrder, stopAtten, Wn, 'high');
                            case '带通'
                                [b, a] = cheby2(filterOrder, stopAtten, Wn, 'bandpass');
                            case '带阻'
                                [b, a] = cheby2(filterOrder, stopAtten, Wn, 'stop');
                        end
                        
                    case '椭圆'
                        switch designType
                            case '低通'
                                [b, a] = ellip(filterOrder, passRipple, stopAtten, Wn, 'low');
                            case '高通'
                                [b, a] = ellip(filterOrder, passRipple, stopAtten, Wn, 'high');
                            case '带通'
                                [b, a] = ellip(filterOrder, passRipple, stopAtten, Wn, 'bandpass');
                            case '带阻'
                                [b, a] = ellip(filterOrder, passRipple, stopAtten, Wn, 'stop');
                        end
                end % 结束switch filterType语句
                
                % 应用滤波器
                % 使用filtfilt进行零相位滤波，避免相位延迟导致波形失真
                try
                    % 对于低阶IIR滤波器，filtfilt通常工作良好
                    if filterOrder <= 8
                        app.iirFilteredAudio = filtfilt(b, a, app.noisyAudio);
                    else
                        % 高阶IIR使用常规filter
                        app.iirFilteredAudio = filter(b, a, app.noisyAudio);
                    end
                catch
                    % 如果filtfilt失败，回退到标准filter
                    app.iirFilteredAudio = filter(b, a, app.noisyAudio);
                end
                
                % 确保输出信号有合理的幅度
                if max(abs(app.iirFilteredAudio)) < 0.01 && max(abs(app.noisyAudio)) > 0.01
                    % 如果滤波后信号幅度太小，恢复到与原始信号相似的幅度
                    scale_factor = max(abs(app.noisyAudio)) / max(abs(app.iirFilteredAudio));
                    app.iirFilteredAudio = app.iirFilteredAudio * scale_factor * 0.8; % 稍微降低以避免削波
                end
                
                % 归一化以防止削波
                if max(abs(app.iirFilteredAudio)) > 1
                    app.iirFilteredAudio = app.iirFilteredAudio / max(abs(app.iirFilteredAudio));
                end
                
                % 绘制滤波后的音频
                plotAudio(app, app.iirFilteredAudio, app.IIRTimeAxes, app.IIRFreqAxes);
                
                % 绘制滤波器频率响应
                plotIIRResponse(app, b, a);
                
            catch ME
                uialert(app.UIFigure, ['滤波器设计错误: ' ME.message], '错误');
            end
        end
        
        % 获取噪声特定的IIR滤波器参数
        function [designType, cutoffFreq, filterOrder, passRipple, stopAtten] = getNoiseSpecificIIRParams(app, noiseType)
            % 默认值
            passRipple = 1;   % 通带波纹(dB)
            stopAtten = 60;   % 阻带衰减(dB)
            
            switch noiseType
                case '高斯白噪声'
                    % 白噪声覆盖宽频带，使用低通滤波器保留语音主要频率
                    designType = '低通';
                    cutoffFreq = 3500;  % 保留0-3.5kHz的主要语音成分
                    filterOrder = 4;     % 降低阶数以确保稳定性
                    
                case '窄带噪声(1000-2000Hz)'
                    % 窄带噪声最适合带阻滤波器
                    designType = '带阻';
                    cutoffFreq = [950 2050];  % 稍宽于噪声带宽
                    filterOrder = 4;           % 带阻通常不需要太高阶数
                    
                case '单频干扰(1500Hz)'
                    % 单频干扰最适合陷波滤波器，实现为窄带带阻
                    designType = '带阻';
                    cutoffFreq = [1450 1550];  % 窄带，集中在1500Hz周围
                    filterOrder = 2;           % 陷波滤波器阶数较低
            end
        end
        
        % 绘制IIR滤波器频率响应的方法
        function plotIIRResponse(app, b, a)
            % 计算频率响应
            [h, w] = freqz(b, a, 1024);
            
            % 绘制幅度响应
            plot(app.IIRResponseAxes, w*app.fs/(2*pi), 20*log10(abs(h)));
            app.IIRResponseAxes.XLabel.String = '频率 (Hz)';
            app.IIRResponseAxes.YLabel.String = '幅度 (dB)';
            
            % 设置标题和Y轴范围
            filterType = app.IIRFilterTypeDropDown.Value;
            designType = app.IIRDesignDropDown.Value;
            
            if ~strcmp(app.IIRNoiseTypeDropDown.Value, '自定义')
                app.IIRResponseAxes.Title.String = [app.IIRNoiseTypeDropDown.Value, ' 专用IIR滤波器响应'];
            else
                app.IIRResponseAxes.Title.String = [filterType, ' ', designType, ' IIR滤波器响应'];
            end
            
            % 根据滤波器类型调整Y轴范围
            if strcmp(designType, '带阻') || contains(designType, '陷波')
                app.IIRResponseAxes.YLim = [-80, 5];
            end
        end
        
        % 应用高级滤波处理的方法
        function applyAdaptiveFiltering(app)
            if isempty(app.noisyAudio)
                uialert(app.UIFigure, '请先向音频添加噪声。', '无噪声音频');
                return;
            end
            
            algorithm = app.AlgorithmDropDown.Value;
            
            switch algorithm
                case 'LMS算法'
                    % 根据噪声类型调整LMS参数
                    if strcmp(app.currentNoiseType, '高斯白噪声')
                        filterOrder = 64;  % 白噪声需要较高阶
                        mu = 0.008;        % 适中的步长
                    elseif strcmp(app.currentNoiseType, '窄带噪声(1000-2000Hz)')
                        filterOrder = 48;  % 中等阶数
                        mu = 0.01;         % 标准步长
                    elseif strcmp(app.currentNoiseType, '单频干扰(1500Hz)')
                        filterOrder = 32;  % 单频较低阶数
                        mu = 0.02;         % 较大步长加快收敛
                    else
                        filterOrder = 32;
                        mu = 0.01;
                    end
                    
                    % 调用自定义LMS算法
                    app.adaptiveAudio = audio_processing('applyLMSFilter', app.noisyAudio, [], mu, filterOrder);
                    
                    titleStr = sprintf('LMS高级滤波处理 (阶数:%d, 步长:%.4f)', filterOrder, mu);
                    
                case '小波去噪'
                    % 应用小波去噪
                    app.adaptiveAudio = audio_processing('applyWaveletDenoising', app.noisyAudio, 'db4', 5);
                    titleStr = '小波去噪结果';
                    
                case '陷波滤波'
                    % 应用陷波滤波器
                    % 根据当前噪声类型确定频率
                    if strcmp(app.currentNoiseType, '单频干扰(1500Hz)')
                        notchFreq = 1500;
                    elseif strcmp(app.currentNoiseType, '窄带噪声(1000-2000Hz)')
                        notchFreq = 1500; % 窄带噪声的中心频率
                    else
                        notchFreq = 1000; % 默认值
                    end
                    app.adaptiveAudio = audio_processing('applyNotchFilter', app.noisyAudio, app.fs, notchFreq, 35);
                    titleStr = '陷波滤波结果';
            end
            
            % 绘制高级滤波处理后的音频
            plotAudio(app, app.adaptiveAudio, app.AdaptiveTimeAxes, app.AdaptiveFreqAxes);
            app.AdaptiveTimeAxes.Title.String = titleStr;
        end
    end
    
    % 处理组件事件的回调函数
    methods (Access = private)
        
        % 按钮按下功能：LoadButton
        function LoadButtonPushed(app, event)
            loadAudio(app);
        end
        
        % 按钮按下功能：PlayOrigButton
        function PlayOrigButtonPushed(app, event)
            if ~isempty(app.originalAudio)
                sound(app.originalAudio, app.fs);
            end
        end
        
        % 按钮按下功能：StopOrigButton
        function StopOrigButtonPushed(app, event)
            clear sound;
        end
        
        % 按钮按下功能：AddNoiseButton
        function AddNoiseButtonPushed(app, event)
            addNoise(app);
        end
        
        % 按钮按下功能：PlayNoisyButton
        function PlayNoisyButtonPushed(app, event)
            if ~isempty(app.noisyAudio)
                sound(app.noisyAudio, app.fs);
            end
        end
        
        % 按钮按下功能：StopNoisyButton
        function StopNoisyButtonPushed(app, event)
            clear sound;
        end
        
        % 按钮按下功能：FilterButton
        function FilterButtonPushed(app, event)
            applyFilter(app);
        end
        
        % 按钮按下功能：PlayFilteredButton
        function PlayFilteredButtonPushed(app, event)
            if ~isempty(app.filteredAudio)
                sound(app.filteredAudio, app.fs);
            end
        end
        
        % 按钮按下功能：StopFilteredButton
        function StopFilteredButtonPushed(app, event)
            clear sound;
        end
        
        % 按钮按下功能：IIRFilterButton
        function IIRFilterButtonPushed(app, event)
            applyIIRFilter(app);
        end
        
        % 按钮按下功能：PlayIIRButton
        function PlayIIRButtonPushed(app, event)
            if ~isempty(app.iirFilteredAudio)
                sound(app.iirFilteredAudio, app.fs);
            end
        end
        
        % 按钮按下功能：StopIIRButton
        function StopIIRButtonPushed(app, event)
            clear sound;
        end
        
        % 按钮按下功能：AdaptiveButton
        function AdaptiveButtonPushed(app, event)
            applyAdaptiveFiltering(app);
        end
        
        % 按钮按下功能：PlayAdaptiveButton
        function PlayAdaptiveButtonPushed(app, event)
            if ~isempty(app.adaptiveAudio)
                sound(app.adaptiveAudio, app.fs);
            end
        end
        
        % 按钮按下功能：StopAdaptiveButton
        function StopAdaptiveButtonPushed(app, event)
            clear sound;
        end
        
        % 添加滤波器类型改变的回调函数
        function FilterTypeChanged(app, event)
            filterType = app.FilterTypeDropDown.Value;
            
            % 更新窗函数下拉框的值
            switch filterType
                case '白噪声专用滤波器'
                    app.WindowTypeDropDown.Value = '汉明窗';
                    app.WindowTypeDropDown.Enable = 'off'; % 锁定窗函数选择
                case '窄带噪声专用滤波器'
                    app.WindowTypeDropDown.Value = '布莱克曼窗';
                    app.WindowTypeDropDown.Enable = 'off'; % 锁定窗函数选择
                case '单频干扰专用滤波器'
                    app.WindowTypeDropDown.Enable = 'off'; % 锁定窗函数选择
                otherwise
                    app.WindowTypeDropDown.Enable = 'on';  % 解锁窗函数选择
            end
        end
        
        % IIR滤波器类型改变回调
        function IIRFilterTypeChanged(app, event)
            filterType = app.IIRFilterTypeDropDown.Value;
            
            % 更新UI控件状态
            switch filterType
                case '巴特沃斯'
                    app.IIRRippleEdit.Enable = 'off';
                    app.IIRAttenuationEdit.Enable = 'off';
                case '切比雪夫I型'
                    app.IIRRippleEdit.Enable = 'on';
                    app.IIRAttenuationEdit.Enable = 'off';
                case '切比雪夫II型'
                    app.IIRRippleEdit.Enable = 'off';
                    app.IIRAttenuationEdit.Enable = 'on';
                case '椭圆'
                    app.IIRRippleEdit.Enable = 'on';
                    app.IIRAttenuationEdit.Enable = 'on';
            end
        end
        
        % IIR滤波器噪声类型改变回调
        function IIRNoiseTypeChanged(app, event)
            noiseType = app.IIRNoiseTypeDropDown.Value;
            
            % 根据预设类型调整界面
            if strcmp(noiseType, '自定义')
                app.IIRFilterTypeDropDown.Enable = 'on';
                app.IIRDesignDropDown.Enable = 'on';
                app.IIROrderEdit.Enable = 'on';
                app.IIRCutoffEdit.Enable = 'on';
                % 根据当前选择的滤波器类型启用/禁用相关参数
                IIRFilterTypeChanged(app, []);
            else
                % 预设类型，锁定参数设置
                app.IIRFilterTypeDropDown.Enable = 'off';
                app.IIRDesignDropDown.Enable = 'off';
                app.IIROrderEdit.Enable = 'off';
                app.IIRCutoffEdit.Enable = 'off';
                app.IIRRippleEdit.Enable = 'off';
                app.IIRAttenuationEdit.Enable = 'off';
                
                % 自动设置滤波器类型
                [designType, cutoffFreq, filterOrder, passRipple, stopAtten] = getNoiseSpecificIIRParams(app, noiseType);
                
                % 更新UI显示
                app.IIRDesignDropDown.Value = designType;
                app.IIROrderEdit.Value = filterOrder;
                app.IIRRippleEdit.Value = passRipple;
                app.IIRAttenuationEdit.Value = stopAtten;
                
                % 显示截止频率
                if isscalar(cutoffFreq)
                    app.IIRCutoffEdit.Value = num2str(cutoffFreq);
                else
                    app.IIRCutoffEdit.Value = [num2str(cutoffFreq(1)), ', ', num2str(cutoffFreq(2))];
                end
                
                % 根据噪声类型自动选择最佳滤波器
                switch noiseType
                    case '高斯白噪声'
                        app.IIRFilterTypeDropDown.Value = '巴特沃斯';
                    case '窄带噪声(1000-2000Hz)'
                        app.IIRFilterTypeDropDown.Value = '椭圆';
                    case '单频干扰(1500Hz)'
                        app.IIRFilterTypeDropDown.Value = '椭圆';
                end
            end
        end
    end

    % 应用初始化和构建
    methods (Access = private)

        % 创建UIFigure和组件
        function createComponents(app)
            % 创建主窗口
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Name = '音频信号处理';
            app.UIFigure.Position = [100, 100, 1000, 700];

            % 创建选项卡组
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.Position = [10, 10, 980, 680];

            % 创建选项卡
            app.OriginalTab = uitab(app.TabGroup);
            app.OriginalTab.Title = '原始音频';
            
            app.NoiseTab = uitab(app.TabGroup);
            app.NoiseTab.Title = '带噪音频';
            
            app.FilteringTab = uitab(app.TabGroup);
            app.FilteringTab.Title = 'FIR滤波器';
            
            app.IIRTab = uitab(app.TabGroup);
            app.IIRTab.Title = 'IIR滤波器';
            
            app.AdaptiveTab = uitab(app.TabGroup);
            app.AdaptiveTab.Title = '高级滤波处理';
            
            % 创建原始音频面板和组件
            app.OriginalPanel = uipanel(app.OriginalTab);
            app.OriginalPanel.Position = [10, 10, 960, 640];
            
            app.LoadButton = uibutton(app.OriginalPanel, 'push');
            app.LoadButton.Position = [20, 600, 100, 30];
            app.LoadButton.Text = '加载音频';
            app.LoadButton.ButtonPushedFcn = createCallbackFcn(app, @LoadButtonPushed, true);
            
            app.PlayOrigButton = uibutton(app.OriginalPanel, 'push');
            app.PlayOrigButton.Position = [130, 600, 100, 30];
            app.PlayOrigButton.Text = '播放';
            app.PlayOrigButton.ButtonPushedFcn = createCallbackFcn(app, @PlayOrigButtonPushed, true);
            
            app.StopOrigButton = uibutton(app.OriginalPanel, 'push');
            app.StopOrigButton.Position = [240, 600, 100, 30];
            app.StopOrigButton.Text = '停止播放';
            app.StopOrigButton.ButtonPushedFcn = createCallbackFcn(app, @StopOrigButtonPushed, true);
            
            app.AudioInfoLabel = uilabel(app.OriginalPanel);
            app.AudioInfoLabel.Position = [350, 600, 590, 30];
            app.AudioInfoLabel.Text = '音频信息';
            
            app.TimeAxes = uiaxes(app.OriginalPanel);
            app.TimeAxes.Position = [20, 320, 920, 270];
            
            app.FreqAxes = uiaxes(app.OriginalPanel);
            app.FreqAxes.Position = [20, 20, 920, 270];
            
            % 创建噪声面板和组件
            app.NoisePanel = uipanel(app.NoiseTab);
            app.NoisePanel.Position = [10, 10, 960, 640];
            
            app.NoiseTypeDropDown = uidropdown(app.NoisePanel);
            app.NoiseTypeDropDown.Items = {'高斯白噪声', '窄带噪声(1000-2000Hz)', '单频干扰(1500Hz)'};
            app.NoiseTypeDropDown.Position = [20, 600, 200, 30];
            app.NoiseTypeDropDown.Value = '高斯白噪声';
            
            app.AddNoiseButton = uibutton(app.NoisePanel, 'push');
            app.AddNoiseButton.Position = [230, 600, 100, 30];
            app.AddNoiseButton.Text = '添加噪声';
            app.AddNoiseButton.ButtonPushedFcn = createCallbackFcn(app, @AddNoiseButtonPushed, true);
            
            app.PlayNoisyButton = uibutton(app.NoisePanel, 'push');
            app.PlayNoisyButton.Position = [340, 600, 100, 30];
            app.PlayNoisyButton.Text = '播放';
            app.PlayNoisyButton.ButtonPushedFcn = createCallbackFcn(app, @PlayNoisyButtonPushed, true);
            
            app.StopNoisyButton = uibutton(app.NoisePanel, 'push');
            app.StopNoisyButton.Position = [450, 600, 100, 30];
            app.StopNoisyButton.Text = '停止播放';
            app.StopNoisyButton.ButtonPushedFcn = createCallbackFcn(app, @StopNoisyButtonPushed, true);
            
            app.NoiseTimeAxes = uiaxes(app.NoisePanel);
            app.NoiseTimeAxes.Position = [20, 320, 920, 270];
            
            app.NoiseFreqAxes = uiaxes(app.NoisePanel);
            app.NoiseFreqAxes.Position = [20, 20, 920, 270];
            
            % 创建滤波面板和组件
            app.FilterPanel = uipanel(app.FilteringTab);
            app.FilterPanel.Position = [10, 10, 960, 640];
            
            app.FilterTypeDropDown = uidropdown(app.FilterPanel);
            app.FilterTypeDropDown.Items = {'低通滤波', '高通滤波', '带阻滤波', ...
                                           '白噪声专用滤波器', '窄带噪声专用滤波器', '单频干扰专用滤波器'};
            app.FilterTypeDropDown.Position = [20, 600, 150, 30];
            app.FilterTypeDropDown.Value = '低通滤波';
            app.FilterTypeDropDown.ValueChangedFcn = createCallbackFcn(app, @FilterTypeChanged, true);
            
            app.WindowTypeDropDown = uidropdown(app.FilterPanel);
            app.WindowTypeDropDown.Items = {'巴特利特窗', '汉宁窗', '汉明窗', '布莱克曼窗', '凯泽窗'};
            app.WindowTypeDropDown.Position = [180, 600, 150, 30];
            app.WindowTypeDropDown.Value = '汉明窗';
            
            app.FilterButton = uibutton(app.FilterPanel, 'push');
            app.FilterButton.Position = [340, 600, 100, 30];
            app.FilterButton.Text = '滤波';
            app.FilterButton.ButtonPushedFcn = createCallbackFcn(app, @FilterButtonPushed, true);
            
            app.PlayFilteredButton = uibutton(app.FilterPanel, 'push');
            app.PlayFilteredButton.Position = [450, 600, 100, 30];
            app.PlayFilteredButton.Text = '播放';
            app.PlayFilteredButton.ButtonPushedFcn = createCallbackFcn(app, @PlayFilteredButtonPushed, true);
            
            app.StopFilteredButton = uibutton(app.FilterPanel, 'push');
            app.StopFilteredButton.Position = [560, 600, 100, 30];
            app.StopFilteredButton.Text = '停止播放';
            app.StopFilteredButton.ButtonPushedFcn = createCallbackFcn(app, @StopFilteredButtonPushed, true);
            
            app.FilterResponseAxes = uiaxes(app.FilterPanel);
            app.FilterResponseAxes.Position = [20, 430, 920, 160];
            
            app.FilteredTimeAxes = uiaxes(app.FilterPanel);
            app.FilteredTimeAxes.Position = [20, 230, 920, 190];
            
            app.FilteredFreqAxes = uiaxes(app.FilterPanel);
            app.FilteredFreqAxes.Position = [20, 20, 920, 190];
            
            % 创建IIR面板和组件
            app.IIRPanel = uipanel(app.IIRTab);
            app.IIRPanel.Position = [10, 10, 960, 640];
            
            % IIR滤波器类型和设计选择
            app.IIRFilterTypeDropDown = uidropdown(app.IIRPanel);
            app.IIRFilterTypeDropDown.Items = {'巴特沃斯', '切比雪夫I型', '切比雪夫II型', '椭圆'};
            app.IIRFilterTypeDropDown.Position = [20, 600, 120, 30];
            app.IIRFilterTypeDropDown.Value = '巴特沃斯';
            app.IIRFilterTypeDropDown.ValueChangedFcn = createCallbackFcn(app, @IIRFilterTypeChanged, true);
            
            app.IIRDesignDropDown = uidropdown(app.IIRPanel);
            app.IIRDesignDropDown.Items = {'低通', '高通', '带通', '带阻'};
            app.IIRDesignDropDown.Position = [150, 600, 120, 30];
            app.IIRDesignDropDown.Value = '低通';
            
            % IIR滤波器参数设置
            uilabel(app.IIRPanel, 'Text', '阶数:', 'Position', [280, 600, 40, 30]);
            app.IIROrderEdit = uieditfield(app.IIRPanel, 'numeric', 'Position', [320, 600, 40, 30], 'Value', 4);
            
            uilabel(app.IIRPanel, 'Text', '截止频率(Hz):', 'Position', [370, 600, 90, 30]);
            app.IIRCutoffEdit = uieditfield(app.IIRPanel, 'text', 'Position', [460, 600, 80, 30], 'Value', '1000');
            
            uilabel(app.IIRPanel, 'Text', '通带波纹(dB):', 'Position', [550, 600, 90, 30]);
            app.IIRRippleEdit = uieditfield(app.IIRPanel, 'numeric', 'Position', [640, 600, 40, 30], 'Value', 1);
            app.IIRRippleEdit.Enable = 'off';
            
            uilabel(app.IIRPanel, 'Text', '阻带衰减(dB):', 'Position', [690, 600, 90, 30]);
            app.IIRAttenuationEdit = uieditfield(app.IIRPanel, 'numeric', 'Position', [780, 600, 40, 30], 'Value', 60);
            app.IIRAttenuationEdit.Enable = 'off';
            
            % 噪声类型预设
            uilabel(app.IIRPanel, 'Text', '噪声类型预设:', 'Position', [20, 560, 90, 30]);
            app.IIRNoiseTypeDropDown = uidropdown(app.IIRPanel);
            app.IIRNoiseTypeDropDown.Items = {'自定义', '高斯白噪声', '窄带噪声(1000-2000Hz)', '单频干扰(1500Hz)'};
            app.IIRNoiseTypeDropDown.Position = [120, 560, 200, 30];
            app.IIRNoiseTypeDropDown.Value = '自定义';
            app.IIRNoiseTypeDropDown.ValueChangedFcn = createCallbackFcn(app, @IIRNoiseTypeChanged, true);
            
            % 操作按钮
            app.IIRFilterButton = uibutton(app.IIRPanel, 'push');
            app.IIRFilterButton.Position = [340, 560, 100, 30];
            app.IIRFilterButton.Text = '应用滤波器';
            app.IIRFilterButton.ButtonPushedFcn = createCallbackFcn(app, @IIRFilterButtonPushed, true);
            
            app.PlayIIRButton = uibutton(app.IIRPanel, 'push');
            app.PlayIIRButton.Position = [450, 560, 100, 30];
            app.PlayIIRButton.Text = '播放';
            app.PlayIIRButton.ButtonPushedFcn = createCallbackFcn(app, @PlayIIRButtonPushed, true);
            
            app.StopIIRButton = uibutton(app.IIRPanel, 'push');
            app.StopIIRButton.Position = [560, 560, 100, 30];
            app.StopIIRButton.Text = '停止播放';
            app.StopIIRButton.ButtonPushedFcn = createCallbackFcn(app, @StopIIRButtonPushed, true);
            
            % 绘图区域
            app.IIRResponseAxes = uiaxes(app.IIRPanel);
            app.IIRResponseAxes.Position = [20, 430, 920, 120];
            app.IIRResponseAxes.Title.String = 'IIR滤波器频率响应';
            
            app.IIRTimeAxes = uiaxes(app.IIRPanel);
            app.IIRTimeAxes.Position = [20, 230, 920, 190];
            app.IIRTimeAxes.Title.String = '时域波形';
            
            app.IIRFreqAxes = uiaxes(app.IIRPanel);
            app.IIRFreqAxes.Position = [20, 20, 920, 190];
            app.IIRFreqAxes.Title.String = '频谱图';
            
            % 创建自适应面板和组件
            app.AdaptivePanel = uipanel(app.AdaptiveTab);
            app.AdaptivePanel.Position = [10, 10, 960, 640];
            
            app.AlgorithmDropDown = uidropdown(app.AdaptivePanel);
            app.AlgorithmDropDown.Items = {'LMS算法', '小波去噪', '陷波滤波'};
            app.AlgorithmDropDown.Position = [20, 600, 200, 30];
            app.AlgorithmDropDown.Value = 'LMS算法';
            
            app.AdaptiveButton = uibutton(app.AdaptivePanel, 'push');
            app.AdaptiveButton.Position = [230, 600, 100, 30];
            app.AdaptiveButton.Text = '应用';
            app.AdaptiveButton.ButtonPushedFcn = createCallbackFcn(app, @AdaptiveButtonPushed, true);
            
            app.PlayAdaptiveButton = uibutton(app.AdaptivePanel, 'push');
            app.PlayAdaptiveButton.Position = [340, 600, 100, 30];
            app.PlayAdaptiveButton.Text = '播放';
            app.PlayAdaptiveButton.ButtonPushedFcn = createCallbackFcn(app, @PlayAdaptiveButtonPushed, true);
            
            app.StopAdaptiveButton = uibutton(app.AdaptivePanel, 'push');
            app.StopAdaptiveButton.Position = [450, 600, 100, 30];
            app.StopAdaptiveButton.Text = '停止播放';
            app.StopAdaptiveButton.ButtonPushedFcn = createCallbackFcn(app, @StopAdaptiveButtonPushed, true);
            
            app.AdaptiveTimeAxes = uiaxes(app.AdaptivePanel);
            app.AdaptiveTimeAxes.Position = [20, 320, 920, 270];
            
            app.AdaptiveFreqAxes = uiaxes(app.AdaptivePanel);
            app.AdaptiveFreqAxes.Position = [20, 20, 920, 270];
        end
    end
    
    methods (Access = public)
        
        % 构造函数
        function app = AudioProcessingApp
            createComponents(app)
            
            % 所有组件创建后显示窗口
            app.UIFigure.Visible = 'on';
        end
        
        % 关闭函数
        function delete(app)
            delete(app.UIFigure);
        end
    end
end