%% 二维Helmholtz方程求解程序  
% 使用四阶有限差分方法  
% 在细网格(2000×2000)和粗网格(1000×1000)上求解并比较  
clear; clc; close all;  
rng(32, 'twister');
%% 参数设置  
% 基本参数  
lambda_val = 10;  
lambda_sq = lambda_val^2;  
c = 0.9 * lambda_val^2;  
num_m =1000;  % 样本数量  

% 定义网格尺寸  
Nx_fine =2000; Ny_fine = 2000;      % 细网格  
Nx_compare =400; Ny_compare = 400;   % 比较网格  

% 计算域范围  
L=1;
x_min = 0; x_max = L;  
y_min = 0; y_max = L;  
% % 傅里叶模式参数  
num_modes2 = 11;  
a =100*pi; b = 110*pi;  
freq_array = linspace(a, b, num_modes2);

% 生成网格点  
[x_fine, y_fine] = meshgrid(linspace(x_min, x_max, Nx_fine+1), linspace(y_min, y_max, Ny_fine+1));  
[x_compare, y_compare] = meshgrid(linspace(x_min, x_max, Nx_compare+1), linspace(y_min, y_max, Nx_compare+1));  

% 预分配结果矩阵（只存储比较网格大小）  
u_matrix = zeros(num_m, Nx_compare+1, Ny_compare+1);  
a_matrix = zeros(num_m, Nx_compare+1, Ny_compare+1);  
f_matrix = zeros(num_m, Nx_compare+1, Ny_compare+1);  
error_matrix = zeros(num_m,1);  
% 时间统计  
total_start_time = tic;  
timing_stats_fine = struct('matrix_build', 0, 'solver', 0);  
timing_stats_coarse = struct('matrix_build', 0, 'solver', 0);  

%% 主循环  
h_wait = waitbar(0, '准备开始计算...');  
for idx=1:num_m
    sample_start_time = tic;  

    % ---- (1) 预先生成所有随机系数 ----  
    t1_start = tic;  
    rand_coeffs = zeros(11, 11, 4); % 存储所有随机系数  
    for m = 0:10  
        for n = 0:10  
            rand_coeffs(m+1,n+1,:) = [(rand()-0.5)*2, (rand()-0.5)*2, ...  
                                     (rand()-0.5)*2, (rand()-0.5)*2];  
        end  
    end  
    omega_rand_scale = rand(); % omega的随机缩放因子  
    t1_time = toc(t1_start);  
    fprintf('步骤1 - 随机系数生成时间: %.4f 秒\n', t1_time);  
    
    % ---- (2) 在细网格上生成场 ----  
    t3_start = tic;   
    omega_fine = zeros(Nx_fine+1, Ny_fine+1);  
    f_2D_fine = zeros(Nx_fine+1, Ny_fine+1);  
    f_fine = zeros(Nx_fine+1, Ny_fine+1);  
    
    add = 100;
    % 使用预生成的随机系数计算omega_fine  
    for m = 0:10  
        m1 = m+add;
        for n = 0:10  
            n1 = n+add;
            omega_fine = omega_fine + ...  
                rand_coeffs(m+1,n+1,1) * cos(m1 * pi * x_fine) .* cos(n1 * pi * y_fine) + ...  
                rand_coeffs(m+1,n+1,2) * sin(m1 * pi * x_fine) .* sin(n1 * pi * y_fine) + ...  
                rand_coeffs(m+1,n+1,3) * cos(m1 * pi * x_fine) .* sin(n1 * pi * y_fine) + ...  
                rand_coeffs(m+1,n+1,4) * sin(m1 * pi * x_fine) .* cos(n1 * pi * y_fine);  
        end  
    end  
    
    % 归一化omega_fine  
    max_omega_val = max(abs(omega_fine(:)));  
    if max_omega_val > 1e-12  
        omega_fine = omega_fine / max_omega_val;  
    end  
    omega_fine = omega_fine * omega_rand_scale;  
    

    % 在两个网格上生成f  
    for i = 1:num_modes2  
        freq = freq_array(i);   
        f_2D_fine = f_2D_fine + (lambda_sq-freq^2-freq^2) * ...  
                       sin(freq * x_fine).* sin(freq * y_fine);  
        f_fine = f_fine + sin(freq * x_fine).* sin(freq * y_fine);      
    end

    % f_2D_fine = f_2D_fine - lambda_sq - c * omega_fine;
    t3_time = toc(t3_start);  
    fprintf('步骤3 - 网格场生成时间: %.4f 秒\n', t3_time);  

    % ---- (4) 分别在细网格和粗网格上求解 ----  
    t4_start = tic;  
    % 细网格求解  
    [u_fine] = solve_helmholtz(omega_fine, f_2D_fine, Nx_fine, Ny_fine, ...  
                              lambda_val, c,x_min,x_max);
    t4_fine_time = toc(t4_start);  
    fprintf('步骤4 - 细网格方程求解时间: %.4f 秒\n', t4_fine_time);  
   
    
    % ---- (5) 插值到比较网格并计算误差 ----  
    t5_start = tic;  
    u_fine_compare = interp2(x_fine, y_fine, u_fine, x_compare, y_compare, 'spline');  
    omega_compare = interp2(x_fine, y_fine, omega_fine, x_compare, y_compare, 'spline');  
    f_compare = interp2(x_fine, y_fine, f_2D_fine, x_compare, y_compare, 'spline');  
   
    % L2_error = norm(u_2D-U_matrix,2)/norm(U_matrix,2);
    % fprintf('L2相对误差: %e\n', L2_error); 
    u_matrix(idx,:,:) = u_fine_compare;  
    a_matrix(idx,:,:) = omega_compare;  
    f_matrix(idx,:,:) = f_compare;  
    t5_time = toc(t5_start);  
    fprintf('步骤5 - 插值时间: %.4f 秒\n', t5_time);  
        
    % 输出每个样本的信息  
    sample_time = toc(sample_start_time);  
    fprintf('样本 %d 完成:\n', idx);   
    fprintf('总计用时: %.4f 秒\n\n', sample_time);  
        
    % 更新进度条  
    waitbar(idx/num_m, h_wait, sprintf('样本 %d/%d (%.1f%%)', ...  
                idx, num_m, idx/num_m*100));  
end  

% 关闭进度条  
close(h_wait);  
% fprintf('一共尝试了%d次生成样本\n', attempts);
% fprintf('一共舍弃了%d个样本\n', discard_samples);
fprintf('一共生成了%d个样本\n', idx);
save('Helmholtz_2D_data.mat', 'u_matrix', 'a_matrix', 'f_matrix', '-v7.3');  

disp('计算完成！结果已保存。');  
