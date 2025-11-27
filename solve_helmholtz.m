% function [u_2D] = solve_helmholtz(omega_2D, f_2D, Nx, Ny, lambda_val, c, timing_stats, x_min, x_max)  
%     % 网格参数 
%     L =x_max-x_min;
%     hx = L/Nx;  
%     % hy = L/Ny;  
%     h2 = hx * hx;  
% 
%     % 系数  
%     coef_center = -60/(12*h2);  
%     coef_adjacent = 16/(12*h2);  
%     coef_far = -1/(12*h2);  
% 
%     % 矩阵构建时间开始  
%     matrix_time = tic;  
% 
%     % 构造离散后的稀疏矩阵A和右端向量F  
%     N_interior = (Nx-1)*(Ny-1);  
% 
%     % 预计算内部点的线性索引矩阵  
%     [I, J] = meshgrid(2:Nx, 2:Ny);  
%     linear_indices = (J-2)*(Nx-1) + (I-1);  
% 
%     % 预分配数组  
%     max_nnz = 13*N_interior - 4*(Nx-1) - 4*(Ny-1);  
%     row_indices = zeros(max_nnz, 1);  
%     col_indices = zeros(max_nnz, 1);  
%     values = zeros(max_nnz, 1);  
% 
%     % 初始化计数器  
%     counter = 1;  
% 
%     % 主对角线元素（中心点）  
%     center_values = coef_center + (lambda_val^2 + c * omega_2D(2:Nx,2:Ny));  
%     center_indices = linear_indices(:);  
%     n_center = numel(center_indices);  
%     row_indices(counter:counter+n_center-1) = center_indices;  
%     col_indices(counter:counter+n_center-1) = center_indices;  
%     values(counter:counter+n_center-1) = center_values(:);  
%     counter = counter + n_center;  
% 
%     % 相邻点（东西南北）  
%     % 西边  
%     valid = I > 2;  
%     idx_west = find(valid);  
%     row_indices(counter:counter+numel(idx_west)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_west)-1) = linear_indices(valid) - 1;  
%     values(counter:counter+numel(idx_west)-1) = coef_adjacent;  
%     counter = counter + numel(idx_west);  
% 
%     % 东边  
%     valid = I < Nx;  
%     idx_east = find(valid);  
%     row_indices(counter:counter+numel(idx_east)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_east)-1) = linear_indices(valid) + 1;  
%     values(counter:counter+numel(idx_east)-1) = coef_adjacent;  
%     counter = counter + numel(idx_east);  
% 
%     % 南边  
%     valid = J > 2;  
%     idx_south = find(valid);  
%     row_indices(counter:counter+numel(idx_south)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_south)-1) = linear_indices(valid) - (Nx-1);  
%     values(counter:counter+numel(idx_south)-1) = coef_adjacent;  
%     counter = counter + numel(idx_south);  
% 
%     % 北边  
%     valid = J < Ny;  
%     idx_north = find(valid);  
%     row_indices(counter:counter+numel(idx_north)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_north)-1) = linear_indices(valid) + (Nx-1);  
%     values(counter:counter+numel(idx_north)-1) = coef_adjacent;  
%     counter = counter + numel(idx_north);  
% 
%     % 远点（四阶格式）  
%     % 西西  
%     valid = I > 3;  
%     idx_westwest = find(valid);  
%     row_indices(counter:counter+numel(idx_westwest)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_westwest)-1) = linear_indices(valid) - 2;  
%     values(counter:counter+numel(idx_westwest)-1) = coef_far;  
%     counter = counter + numel(idx_westwest);  
% 
%     % 东东  
%     valid = I < Nx-1;  
%     idx_easteast = find(valid);  
%     row_indices(counter:counter+numel(idx_easteast)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_easteast)-1) = linear_indices(valid) + 2;  
%     values(counter:counter+numel(idx_easteast)-1) = coef_far;  
%     counter = counter + numel(idx_easteast);  
% 
%     % 南南  
%     valid = J > 3;  
%     idx_southsouth = find(valid);  
%     row_indices(counter:counter+numel(idx_southsouth)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_southsouth)-1) = linear_indices(valid) - 2*(Nx-1);  
%     values(counter:counter+numel(idx_southsouth)-1) = coef_far;  
%     counter = counter + numel(idx_southsouth);  
% 
%     % 北北  
%     valid = J < Ny-1;  
%     idx_northnorth = find(valid);  
%     row_indices(counter:counter+numel(idx_northnorth)-1) = linear_indices(valid);  
%     col_indices(counter:counter+numel(idx_northnorth)-1) = linear_indices(valid) + 2*(Nx-1);  
%     values(counter:counter+numel(idx_northnorth)-1) = coef_far;  
%     counter = counter + numel(idx_northnorth);  
% 
%     % 构建最终的稀疏矩阵  
%     A = sparse(row_indices(1:counter-1), col_indices(1:counter-1), values(1:counter-1), N_interior, N_interior);  
% 
%     % 构建右端向量  
%     F = f_2D(2:Nx,2:Ny);  
%     F = F(:);  
% 
%     % 记录矩阵构建时间  
%     timing_stats.matrix_build = timing_stats.matrix_build + toc(matrix_time);  
% 
%     % 求解器时间开始  
%     solver_time = tic;  
% 
%     % 求解线性方程组  
%     u_interior = A \ F;  
% 
%     % 记录求解时间  
%     timing_stats.solver = timing_stats.solver + toc(solver_time);  
% 
%     % 填回解到网格  
%     u_2D = zeros(Nx+1, Ny+1);  
%     lin_id = @(i,j) (j-2)*(Nx-1) + (i-1);  
%     for j = 2:Ny  
%         for i = 2:Nx  
%             row_id = lin_id(i,j);  
%             u_2D(i,j) = u_interior(row_id);  
%         end  
%     end  
% end  

function [u_2D] = solve_helmholtz(omega_2D, f_2D, Nx, Ny, lambda_val, c, x_min, x_max)  
    % 求解2D Helmholtz方程  
    %  
    % 输入参数:  
    %   omega_2D: 介质分布矩阵  
    %   f_2D: 右端项  
    %   Nx, Ny: x和y方向的网格点数  
    %   lambda_val: lambda参数值  
    %   c: 介质系数  
    %   x_min, x_max: 计算域范围  
    %  
    % 输出参数:  
    %   u_2D: 数值解  

    % 计算网格间距  
    hx = (x_max - x_min) / Nx;  
    hy = (x_max - x_min) / Ny;  % 假设y方向范围相同  

    % 定义线性索引函数（局部函数）  
    function id = lin_id(i, j)  
        id = (j-2)*(Nx-1) + (i-1);  
    end  
    
    % 内部点数量  
    N_interior = (Nx-1)*(Ny-1);  
    
    % 构建主对角线  
    omega_interior = reshape(omega_2D(2:Nx,2:Ny), [], 1);  
    main_diag = -2/(hx^2) - 2/(hy^2) + (lambda_val^2 + c * omega_interior);  
    
    % x方向的次对角线  
    x_offdiag = ones(N_interior-1, 1) / hx^2;  
    x_offdiag(Nx-1:Nx-1:end) = 0;  
    
    % y方向的次对角线  
    y_offdiag = ones(N_interior-(Nx-1), 1) / hy^2;  
    
    % 构建稀疏矩阵  
    A = sparse(1:N_interior, 1:N_interior, main_diag, N_interior, N_interior) + ...  
        sparse(2:N_interior, 1:N_interior-1, x_offdiag, N_interior, N_interior) + ...  
        sparse(1:N_interior-1, 2:N_interior, x_offdiag, N_interior, N_interior) + ...  
        sparse(1:N_interior-(Nx-1), (Nx-1)+1:N_interior, y_offdiag, N_interior, N_interior) + ...  
        sparse((Nx-1)+1:N_interior, 1:N_interior-(Nx-1), y_offdiag, N_interior, N_interior);  
    
    % 右端向量  
    F = reshape(f_2D(2:Nx,2:Ny), [], 1);  
    
    % 求解线性方程组  
    u_interior = A \ F;  
    
    % 将解填回到完整网格  
    u_2D = zeros(Nx+1, Ny+1);  
    for j = 2:Ny  
        for i = 2:Nx  
            row_id = lin_id(i,j);  
            u_2D(i,j) = u_interior(row_id);  
        end  
    end  
end  