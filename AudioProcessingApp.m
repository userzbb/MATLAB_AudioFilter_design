% filepath: g:\zizim\Documents\code\matlab_project\demo_2\AudioProcessingApp.m
classdef AudioProcessingApp < matlab.apps.AppBase

    % 对应于应用组件的属性
    properties (Access = public)
        UIFigure           matlab.ui.Figure
        TabGroup           matlab.ui.container.TabGroup
        OriginalTab        matlab.ui.container.Tab
        NoiseTab           matlab.ui.container.Tab
        FilteringTab       matlab.ui.container.Tab
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
        
        % 滤波面板
        FilterPanel        matlab.ui.container.Panel
        FilterTypeDropDown matlab.ui.control.DropDown
        WindowTypeDropDown matlab.ui.control.DropDown
        FilterButton       matlab.ui.control.Button
        FilterResponseAxes matlab.ui.control.UIAxes
        FilteredTimeAxes   matlab.ui.control.UIAxes
        FilteredFreqAxes   matlab.ui.control.UIAxes
        PlayFilteredButton matlab.ui.control.Button
        StopFilteredButton matlab.ui.control.Button
        
        % 自适应面板
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
        adaptiveAudio      % 自适应滤波后的音频
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
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'low', cutoff, windowType);
                    
                case '高通滤波'
                    % 作为演示，使用高通滤波器
                    cutoff = 2100; % 高于噪声频率
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'high', cutoff, windowType);
                    
                case '带阻滤波'
                    if strcmp(app.currentNoiseType, '窄带噪声(1000-2000Hz)')
                        f1 = 950; f2 = 2050; % 略宽于噪声频带
                    else % 单频干扰
                        f1 = 1450; f2 = 1550; % 围绕1500Hz
                    end
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'stop', [f1 f2], windowType);
                    
                case '白噪声专用滤波器'
                    % 为白噪声设计的优化滤波器 - 使用低通滤波
                    % 语音信号大部分能量集中在4kHz以下
                    cutoff = 3500; % 略低于标准低通以提供更好的噪声抑制
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'low', cutoff, '汉明窗');
                    
                case '窄带噪声专用滤波器'
                    % 为1000-2000Hz窄带噪声设计的专用滤波器
                    % 使用精确调整的带阻滤波器
                    f1 = 980; f2 = 2020; % 精确匹配噪声带宽
                    app.filteredAudio = audio_processing('applyFIRFilter', app.noisyAudio, app.fs, 'stop', [f1 f2], '布莱克曼窗');
                    
                case '单频干扰专用滤波器'
                    % 为1500Hz单频干扰设计的专用滤波器
                    % 使用陷波滤波器，最适合单频干扰
                    app.filteredAudio = audio_processing('applyNotchFilter', app.noisyAudio, app.fs, 1500, 35);
            end
            
            % 绘制滤波后的音频
            plotAudio(app, app.filteredAudio, app.FilteredTimeAxes, app.FilteredFreqAxes);
            
            % 绘制滤波器频率响应
            plotFilterResponse(app, filterType, windowType);
        end
        
        % 绘制滤波器频率响应的方法
        function plotFilterResponse(app, filterType, windowType)
            % 根据当前设置创建滤波器
            switch filterType
                case '低通滤波'
                    cutoff = 4000; % 示例值，应与applyFilter中使用的值匹配
                    [h, w] = audio_processing('getFilterResponse', 'low', cutoff, app.fs, windowType);
                    titleStr = '低通滤波器频率响应';
                    
                case '高通滤波'
                    cutoff = 2100;
                    [h, w] = audio_processing('getFilterResponse', 'high', cutoff, app.fs, windowType);
                    titleStr = '高通滤波器频率响应';
                    
                case '带阻滤波'
                    f1 = 950; f2 = 2050; % 示例，应与applyFilter匹配
                    [h, w] = audio_processing('getFilterResponse', 'stop', [f1 f2], app.fs, windowType);
                    titleStr = '带阻滤波器频率响应';
                    
                case '白噪声专用滤波器'
                    cutoff = 3500;
                    [h, w] = audio_processing('getFilterResponse', 'low', cutoff, app.fs, '汉明窗');
                    titleStr = '白噪声专用滤波器频率响应';
                    
                case '窄带噪声专用滤波器'
                    f1 = 980; f2 = 2020;
                    [h, w] = audio_processing('getFilterResponse', 'stop', [f1 f2], app.fs, '布莱克曼窗');
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
        
        % 应用自适应滤波的方法
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
                    
                    titleStr = sprintf('LMS自适应滤波 (阶数:%d, 步长:%.4f)', filterOrder, mu);
                    
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
            
            % 绘制自适应滤波后的音频
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
            app.FilteringTab.Title = '滤波处理';
            
            app.AdaptiveTab = uitab(app.TabGroup);
            app.AdaptiveTab.Title = '自适应滤波';
            
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