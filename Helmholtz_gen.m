% 清除环境  
clear; clc; close all;  
% 参数设置  
lambda_val = 10;  
c = 0.9 * lambda_val^2;  % 3.6  
num_m =10;  % 生成的样本数量  
num_modes2 = 11;  % 正弦模式的数量 
a = 300; %  定义区间下界 a  
b =350; %  定义区间上界 b  
freq_array = linspace(a, b, num_modes2); % 生成等距采样数组 

% 网格划分  
a_domain = -1;  
b_domain = 1;  
N = 10000;
h = (b_domain - a_domain) / N;  % 网格间距  
x = linspace(a_domain, b_domain, N+1)';  % 网格点位置 (列向量)  
% 预分配存储矩阵  
new_a_matrix = zeros(num_m, N+1);  % 存储 ω(x)  
new_u_matrix = zeros(num_m, N+1);  % 存储 u(x)  
new_f_matrix = zeros(num_m, N+1);  % 存储 u(x)  

% 预计算常数  
lambda_sq = lambda_val^2;  

% 显示进度条  
h_wait = waitbar(0, '开始计算...');  


for idx = 1:num_m 
     
    % 更新进度条每100次  
    if mod(idx, 1) == 0  
        waitbar(idx / num_m, h_wait, sprintf('计算进度: %d / %d', idx, num_m));  
    end  

    % % 生成 omega(x)
    omega = zeros(N+1, 1);  


    for n = 10:51
        omega = omega + (rand()-0.5)* 2 * cos((n-1)*pi * x)+ (rand()-0.5)* 2 * sin((n-1)*pi * x); 
        % 归一化 omega   
        omega = omega/max(abs(omega));
        omega = omega * rand();
    end

    f = zeros(N+1, 1); 
    f1 = zeros(N+1, 1);
    for n = 1:num_modes2
        freq = freq_array(n);
        f = f + (lambda_sq-freq^2) * sin(freq * x) ;
        f1 = f1 + sin(freq * x);
    end
     

    % 构建系数矩阵 A 和右侧向量 F  
    % 由于边界条件 u0 = uN = 0，方程仅适用于内点 i = 2 到 N  
    tic;
    % 构建三对角矩阵的高效方法  
    main_diag = -2/h^2 + (lambda_sq + c * omega(2:N));  % 主对角线  
    off_diag = ones(N-2, 1) / h^2;  % 次对角线  

    % 构建稀疏矩阵   
    A = sparse(1:N-1, 1:N-1, main_diag, N-1, N-1) + ...  
        sparse(2:N-1, 1:N-2, off_diag, N-1, N-1) + ...  
        sparse(1:N-2, 2:N-1, off_diag, N-1, N-1);  

    % 右侧向量  
    F_vec = f(2:N);

    % 求解线性方程组 A * u_inner = F_vec  
    u_inner = A \ F_vec;  
    % 添加边界条件，构建完整的解向量 u  
    u = zeros(N+1, 1);  
    u(2:N) = u_inner; 
    
    % u = u + 1;
    % 结束计时并计算运行时间  
    elapsed_time = toc;  

    % 输出运行时长  
    fprintf('模型推理时间：%.6f 秒\n', elapsed_time);  
    % for i = 1:M
    %     u = u+A_M(i)*cos(i*omega) +B_M(i)*sin(i*omega);
    % end
    %生成一般非线性算子
    % u = sin(30*omega);
    % % 存储 omega(x) 和 u(x) 到矩阵  
    % L2_error = norm((u-f1),2)/norm(f1,2);
    % fprintf('L2相对误差: %e\n', L2_error); 

    % 存储 omega(x) 和 u(x) 到矩阵  
    new_a_matrix(idx, :) = omega';  
    new_u_matrix(idx, :) = u';  
    new_f_matrix(idx, :) = f1';
    L2_error = norm((u-f1),2)/norm(f1,2);
    fprintf('L2相对误差: %e\n', L2_error); 
end  

% 关闭进度条  
close(h_wait);  
% 保存结果到一个 .mat 文件  
a_matrix = new_a_matrix;
u_matrix = new_u_matrix;
f_matrix = new_f_matrix;
% save('Helmholta_data.mat', 'a_matrix', 'u_matrix','f_matrix', '-v7.3');  

% 提示完成  
disp('所有计算完成，结果已保存到 Helmholta_data0.mat 中。');

