% 启动音频处理应用程序
% 此脚本用于启动打包后的音频处理应用程序

% 清空工作区和命令窗口
clear all;
clc;

% 检查必要的工具箱是否已安装
required_toolboxes = {'Signal_Toolbox', 'Wavelet_Toolbox', 'DSP_System_Toolbox'};
missing_toolboxes = {};

installed_toolboxes = ver;
installed_names = {installed_toolboxes.Name};

for i = 1:length(required_toolboxes)
    found = false;
    for j = 1:length(installed_names)
        if contains(installed_names{j}, required_toolboxes{i})
            found = true;
            break;
        end
    end
    if ~found
        missing_toolboxes{end+1} = required_toolboxes{i};
    end
end

if ~isempty(missing_toolboxes)
    warning('以下工具箱可能未安装，应用可能无法正常运行:');
    disp(missing_toolboxes);
end

% 添加当前脚本所在目录到MATLAB路径，以确保AudioProcessingApp可被找到
% 获取当前脚本所在目录
script_dir = fileparts(mfilename('fullpath'));
% 添加该目录到MATLAB路径
addpath(script_dir);
% 打印信息 (用于确认路径已添加)
fprintf('信息: 已将目录 "%s" 添加到MATLAB路径。\n', script_dir);

% 启动应用程序
fprintf('正在启动音频处理应用程序...\n');

    % 尝试启动应用程序
    app = AudioProcessingApp;
    fprintf('应用程序已成功启动!\n');

