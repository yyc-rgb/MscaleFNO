% 清除环境  
clear; clc; close all;  
% load("Helmholta_data_com_Orthogonal_w50_norm_-11_rand_L0pi_U50pi_1strandf11_L0pi_U10pi_fixed_lambda10_res10000_-1_1.mat")
% total_samples = size(u_matrix, 1);  
% sample_idx = 1;  
% A_matrix = a_matrix(sample_idx, :)';  
% U_matrix = u_matrix(sample_idx, :)'; 
% sample_idx2 = randi(total_samples);  
% A_matrix2 = a_matrix(sample_idx2, :)';  
% U_matrix2 = u_matrix(sample_idx2, :)'; 

% L_2error = norm(U_matrix2-U_matrix,2)/norm(U_matrix,2);
% figure;  % 创建新图窗  
% plot(U_matrix-U_matrix2, 'LineWidth', 2, 'Color', 'b', 'DisplayName', 'Solution 1');  % 绘制第一个解  
% hold on;  % 保持图形  
% plot(U_matrix2, 'LineWidth', 2, 'Color', 'r', '--', 'DisplayName', 'Solution 2');  % 绘制第二个解  
% 
% % 添加图例、标题和标签  
% legend('show');  % 显示图例  
% title('Comparison of Two Solutions');  
% xlabel('Index');  
% ylabel('Value');  
% grid on;  % 添加网格  
% set(gca, 'FontSize', 12);  % 设置坐标轴字体大小  


% 参数设置  
lambda_val = 10;  
c = 0.9 * lambda_val^2;  % 3.6  
num_m =10;  % 生成的样本数量  
num_modes2 = 11;  % 正弦模式的数量 
a = 60*pi; %  定义区间下界 a  
b =70*pi; %  定义区间上界 b  
freq_array = linspace(a, b, num_modes2); % 生成等距采样数组 
% D_M = zeros(num_modes2,1);
% for i = 1:num_modes2
%     D_M(i) = (rand()-0.5)*2;
% end
% M = 40;
% A_M = zeros(M,1);
% B_M = zeros(M,1);
% for i = 1:M
%     A_M(i) = (rand()-0.5)*2;
%     B_M(i) = (rand()-0.5)*2;
% end

% 网格划分  
a_domain = -1;  
b_domain = 1;  
N = 10000;
% step = N/1000;
h = (b_domain - a_domain) / N;  % 网格间距  
x = linspace(a_domain, b_domain, N+1)';  % 网格点位置 (列向量)  
% 预分配存储矩阵  
new_a_matrix = zeros(num_m, N+1);  % 存储 ω(x)  
new_u_matrix = zeros(num_m, N+1);  % 存储 u(x)  
new_f_matrix = zeros(num_m, N+1);  % 存储 u(x)  

% m = 100;
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
    % for n = 1:10
    %         % 累加到 omega(x)  
    %         omega = omega + (rand()-0.5)* 2 * cos((n-1)*pi * x)+ (rand()-0.5)* 2 * sin((n-1)*pi * x); 
    % end
    %     % omega = omega + x.^2 -1;
    %     % 归一化 omega   
    %     omega = omega/max(abs(omega));
    %     omega = omega * 0.2;

    if idx>=1
        % omega = new_a_matrix(idx-1, :)'+  0.8*cos(15*pi * x);
        a1 = 1;
        b1 = a1+80;
        for n = a1:b1
            % 累加到 omega(x)  
            omega = omega + (rand()-0.5)* 2 * cos((n-1)*pi * x)+ (rand()-0.5)* 2 * sin((n-1)*pi * x); 
        end
        % 归一化 omega   
        omega = omega/max(abs(omega));
        omega = omega * rand();
    end

    % omega = omega + a_matrix(idx, :)';
    % 
    f = zeros(N+1, 1); 
    f1 = zeros(N+1, 1);
    for n = 1:num_modes2
        % 累加到 f(x)
        % d = D_M(n);
        d=1.0;
        freq = freq_array(n);
        f = f + (lambda_sq-freq^2) * sin(freq * x) * d;
        f1 = f1 + sin(freq * x)*d;
    end
     
    % f = f - (lambda_sq+c*omega);

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
% load('omega_a_com_norm_-11_L0_U10pi_u_Sum_rand_com(M_omega)_ML1_MU10_data.mat', 'a_matrix', 'u_matrix');
% a_matrix = [a_matrix; new_a_matrix];
% u_matrix = [u_matrix; new_u_matrix];

% 保存结果到一个 .mat 文件  
a_matrix = new_a_matrix;
u_matrix = new_u_matrix;
f_matrix = new_f_matrix;
% save('Helmholta_data_com_Orthogonal_w50_norm_-11_rand_L0pi_U50pi_f10_L0pi_U10pi_Bound0_lambda50_res10000_-1_1.mat', 'a_matrix', 'u_matrix','f_matrix', '-v7.3');  
% save('Helmholta_data_com_Orthogonal_w5_norm_-11_rand_L0pi_U10pi_f0_Bound1_lambda10_res10000_-1_1.mat', 'a_matrix', 'u_matrix','f_matrix', '-v7.3');  
% save('data_check.mat', 'a_matrix', 'u_matrix','f_matrix');  
% L_2error = norm(squeeze(u_matrix)-squeeze(f_matrix(1,:)),2)/norm(squeeze(f_matrix(1,:)),2);
% L_2error = norm(u_matrix(1,:)'-U_matrix,2)/norm(U_matrix,2);
% save('omega_1d_generalization.mat', 'a_matrix', 'u_matrix');

% 提示完成  
disp('所有计算完成，结果已保存到 Helmholta_data0.mat 中。');

 % % 方法2：使用spdiags  
    % % 主对角线(长度N-1)  
    % d = -2/h^2 + (lambda_sq + c * omega(2:N));    
    % 
    % % 次对角线(长度N-2)和补零  
    % e = ones(N-2,1)/h^2;                          
    % e_lower = [e; 0];        % 下对角线补零到N-1长度  
    % e_upper = [0; e];        % 上对角线补零到N-1长度  
    % 
    % % 构建对角线矩阵(每列长度必须为N-1)  
    % B = [e_lower d e_upper];   % (N-1)×3的矩阵  
    % 
    % % 构建三对角矩阵  
    % A = spdiags(B, [-1 0 1], N-1, N-1); 
    % F_vec = f(2:N);
    % 
    % % 求解线性方程组 A * u_inner = F_vec  
    % u_inner = A \ F_vec;  
    % % 添加边界条件，构建完整的解向量 u  
    % u = zeros(N+1, 1);  
    % u(2:N) = u_inner; 
    
