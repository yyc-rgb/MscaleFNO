% 参数设置  
% clear; clc; close all;  
% load("Helmholtz_2D_2000to2000_w10_L0_U10pi_f36_L200_U250_lambda10_L1_10-4_1.mat")
% total_samples = size(u_matrix, 1);  
% sample_idx = randi(total_samples);  
% A_matrix = squeeze(a_matrix(sample_idx, :, :));  
% U_matrix = squeeze(u_matrix(sample_idx, :, :)); 

lambda_val = 10;  
lambda_sq = lambda_val^2;  
c = 0.9 * lambda_val^2;   
num_m = 10;       % 生成样本数量(示例10个)  
Nx = 1000;          % x方向网格分段  
Ny = 1000;          % y方向网格分段  
L=1;
x_min = 0; x_max = L;  
y_min = 0; y_max = L;  
hx = (x_max - x_min) / Nx;  
hy = (y_max - y_min) / Ny;  
[x, y] = meshgrid(linspace(x_min, x_max, Nx+1), linspace(y_min, y_max, Ny+1));  % 二维网格点位置  

num_modes2 = 10;  % 正弦模式的数量 
a = 50; %  定义区间下界 a  
b = 60; %  定义区间上界 b  
freq_array = linspace(a, b, num_modes2); % 生成等距采样数组 

% 预分配3D矩阵: 大小 (num_m, Nx+1, Ny+1)   
a_matrix = zeros(num_m, Nx+1, Ny+1);  
u_matrix = zeros(num_m, Nx+1, Ny+1);  
f_matrix = zeros(num_m, Nx+1, Ny+1); 
%% 显示进度  
h_wait = waitbar(0,'准备开始计算...');  

for idx = 1:num_m  
    % 更新进度条  
    if mod(idx,1)==0  
        waitbar(idx/num_m, h_wait, ...  
            sprintf('正在计算第 %d / %d 个样本...', idx, num_m));  
    end  
    
    % ---- (1) 构造随机 ω(x,y) 和 f(x,y) ----  
    % 这里演示用若干个随机正弦/余弦叠加，细节可按需修改  
    omega_2D = zeros(Nx+1, Ny+1);  
    f_2D     = ones(Nx+1, Ny+1);  
    f     = ones(Nx+1, Ny+1);  
    
    for m = 0:20
        for n = 0:20
            % 使用傅里叶正交基函数生成 omega(x, y)
            omega_2D = omega_2D + (rand() - 0.5) * 2 * cos(2 * m * pi * x) .* cos(2 * n * pi * y ) ...
                            + (rand() - 0.5) * 2 * sin(2 * m * pi * x ) .* sin(2 * n * pi * y)...
                            + (rand() - 0.5) * 2 * cos(2 * m * pi * x ) .* sin(2 * n * pi * y)...
                            + (rand() - 0.5) * 2 * sin(2 * m * pi * x ) .* cos(2 * n * pi * y);
        end
    end 

    % 归一化 omega  
    max_omega_val = max(abs(omega_2D(:)));  
    if max_omega_val > 1e-12  
        omega_2D = omega_2D / max_omega_val;  
    end

    omega_2D = omega_2D *rand();

    % omega_2D = omega_2D + A_matrix;  
    for i = 1:num_modes2
        freq = freq_array(i);
        f_2D = f_2D + (lambda_sq-freq^2-freq^2) * sin(freq * x).* sin(freq * y); 
        f= f + sin(freq * x).* sin(freq * y); 
    end  

    % for i = 1:num_modes2
    %     freq_x = freq_array(i);
    %     for j = 1:num_modes2  
    %         freq_y = freq_array(j);
    %         f_2D = f_2D + (lambda_sq-freq_x^2-freq_y^2) * sin(freq_x * x).* sin(freq_y * y); 
    %         f= f + sin(freq_x * x).* sin(freq_y * y); 
    %     end  
    % end  
    f_2D = f_2D / (num_modes2^2);
    
    % f_2D = f_2D - (lambda_sq+c*omega_2D);

    % ---- (2) 构造离散后的稀疏矩阵A和右端向量F ----
    % 填表函数: 将(i,j)转换为线性索引  
    % 注意: i,j是内部点(2..Nx, 2..Ny)  
    lin_id = @(i,j) (j-2)*(Nx-1) + (i-1);
    % 内部点数量  
    N_interior = (Nx-1)*(Ny-1);  

    % 构建主对角线  
    % 重塑omega_2D为一维向量（只取内部点）  
    omega_interior = reshape(omega_2D(2:Nx,2:Ny), [], 1);  
    main_diag = -2/(hx^2) - 2/(hy^2) + (lambda_val^2 + c * omega_interior);  

    % x方向的次对角线（1/hx^2）  
    % 需要在每行内部点之间连接  
    x_offdiag = ones(N_interior-1, 1) / hx^2;  
    % 移除行之间的连接  
    x_offdiag(Nx-1:Nx-1:end) = 0;  

    % y方向的次对角线（1/hy^2）  
    % 连接相邻行的对应点  
    y_offdiag = ones(N_interior-(Nx-1), 1) / hy^2;  

    % 构建稀疏矩阵  
    A = sparse(1:N_interior, 1:N_interior, main_diag, N_interior, N_interior) + ...      
        sparse(2:N_interior, 1:N_interior-1, x_offdiag, N_interior, N_interior) + ...  
        sparse(1:N_interior-1, 2:N_interior, x_offdiag, N_interior, N_interior) + ...  
        sparse(1:N_interior-(Nx-1), (Nx-1)+1:N_interior, y_offdiag, N_interior, N_interior) + ...  
        sparse((Nx-1)+1:N_interior, 1:N_interior-(Nx-1), y_offdiag, N_interior, N_interior);  

    % 右端向量  
    F = reshape(f_2D(2:Nx,2:Ny), [], 1); 
    % ---- (3) 求解线性方程组 A * u_interior = F ----  
    % 这里的 u_interior 是去掉边界(Dirichlet=0)的内部未知量  
    u_interior = A \ F;  % 直接求解(可换成更高效的迭代方法)  

    % ---- (4) 将解填回到 (Nx+1)×(Ny+1) 的网格上 ----  
    u_2D = zeros(Nx+1, Ny+1);  
    for j = 2:Ny  
        for i = 2:Nx  
            row_id = lin_id(i,j);  
            u_2D(i,j) = u_interior(row_id);  
        end  
    end   
    % u_2D = u_2D +1;
    % L2_error = norm(u_2D-f,2)/norm(f,2);
    % L2_error = norm(u_2D-U_matrix,2)/norm(U_matrix,2);
    % fprintf('L2相对误差: %e\n', L2_error); 
    % ---- (5) 存储到3D数组中 ----  
    a_matrix(idx,:,:) = omega_2D;   
    u_matrix(idx,:,:) = u_2D;  
    f_matrix(idx,:,:) = f; 
end  
close(h_wait);  
disp('二维Helmholtz方程的有限差分样本已生成。'); 
% save('Helmholtz_2D_lambda10_omega_5.mat', 'a_matrix', 'u_matrix','f_matrix'); 


   % % 周期边界条件  
    % N_total = (Nx+1) * (Ny+1);  % 修正总点数  
    % 
    % % 线性索引转换  
    % lin_id = @(i,j) (j-1)*(Nx+1) + i;  
    % 
    % % 构建矩阵元素  
    % omega_all = reshape(omega_2D, [], 1);  
    % main_diag = -2/(hx^2) - 2/(hy^2) + (lambda_val^2 + c * omega_all);  
    % 
    % % x方向连接  
    % x_offdiag = ones(N_total-1, 1) / hx^2;  
    % x_offdiag(Nx+1:Nx+1:end) = 0;  % 移除行间普通连接  
    % 
    % % y方向连接  
    % y_offdiag = ones(N_total-(Nx+1), 1) / hy^2;  
    % 
    % % 周期边界连接  
    % x_periodic = sparse(lin_id(1,1:Ny+1), lin_id(Nx+1,1:Ny+1), 1/hx^2, N_total, N_total);  
    % y_periodic = sparse(lin_id(1:Nx+1,1), lin_id(1:Nx+1,Ny+1), 1/hy^2, N_total, N_total);  
    % 
    % % 构建完整矩阵  
    % A = sparse(1:N_total, 1:N_total, main_diag, N_total, N_total) + ...  
    %     sparse(2:N_total, 1:N_total-1, x_offdiag, N_total, N_total) + ...  
    %     sparse(1:N_total-1, 2:N_total, x_offdiag, N_total, N_total) + ...  
    %     sparse(1:N_total-(Nx+1), (Nx+1)+1:N_total, y_offdiag, N_total, N_total) + ...  
    %     sparse((Nx+1)+1:N_total, 1:N_total-(Nx+1), y_offdiag, N_total, N_total) + ...  
    %     x_periodic + x_periodic' + ...  
    %     y_periodic + y_periodic';  
    % 
    % % 右端项  
    % F = reshape(f_2D, [], 1);  
    % 
    % % 求解线性系统  
    % u_solution = A \ F;  
    % 
    % % 重构为二维数组  
    % u_2D = reshape(u_solution, Nx+1, Ny+1);  
