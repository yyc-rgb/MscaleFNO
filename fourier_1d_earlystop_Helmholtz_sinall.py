import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from torchinfo import summary
import os
import csv  

import torch
import numpy as np
import random

def set_seed(seed):
    """设置随机数种子以确保参数初始化一致"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保使用确定性算法（某些情况下可能会稍慢）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 在运行代码的最开始设置随机数种子
set_seed(20)  # 您可以将 42 替换为任意固定的整数

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 4 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def sfm_activation(self, z, s=0.5):
        """ SFM 激活函数，输出是两个分量的和 """
        # return s * (torch.cos(z) + torch.sin(z))  # 求和
        return torch.sin(z)  # 求和


    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        
        
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [self.padding,self.padding]) # pad the domain if input is non-periodic
        
        
        # x = self.sfm_activation(x)
        
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.sfm_activation(x)  # 使用 SFM 激活函数，输出是两个分量的和


        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.sfm_activation(x)
        
        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.sfm_activation(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., self.padding:-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

class MultiScaleFNO1d(nn.Module):
    def __init__(self, modes, width, num_subnets=8):
        super(MultiScaleFNO1d, self).__init__()
        # self.width = width
        # self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.num_subnets = num_subnets
        self.subnets = nn.ModuleList([FNO1d(modes, width) for _ in range(num_subnets)])
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0, 80.0, 160.0, 200.0, 240.0, 280.0, 360.0, 400.0]))  # 可训练的缩放因子
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0, 16.0, 32.0, 40.0, 48.0, 56.0, 72.0, 80.0]))  # 可训练的缩放因子
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0]))  # 可训练的缩放因子
        self.scaling_factors = nn.Parameter(torch.tensor([1.0, 40.0, 80.0, 100.0, 120.0, 140.0, 180.0, 200.0]))  # 可训练的缩放因子
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))  # 可训练的缩放因子
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0,10.0,20.0,40.0,60.0,80.0,100.0,120.0]))  # 可训练的缩放因子
        # self.weights = nn.Parameter(torch.tensor([1.0,10.0,20.0,40.0,60.0,80.0,100.0,120.0]))  # 每个子网络的权重
        # self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))  # 每个子网络的权重
        self.weights = nn.Parameter(torch.tensor([1.0/num_subnets]*num_subnets))  # 每个子网络的权重
        # self.weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]))  # 每个子网络的权重
        # self.scaling_factors = torch.tensor([100.0]).cuda()  
        # self.weights = torch.tensor([1.0]).cuda()
    def forward(self, x):
        outputs = []
        
        for i, net in enumerate(self.subnets):
            # 获取缩放后的输入，保持与FNO1d相同的输入格式
            grid = self.get_scaled_grid(x.shape, x.device, scale=self.scaling_factors[i])
            x_scaled = torch.cat((x*self.scaling_factors[i], grid), dim=-1)
            # x_scaled = torch.cat(((self.scaling_factors[i]+(self.scaling_factors[i]-1.0)/2 )*x, grid), dim=-1)
            outputs.append(net(x_scaled))
        

        # 将每个子网络的输出加权求和
        weighted_sum = torch.stack(outputs, dim=0)
        weighted_sum = torch.einsum('i,i...->...', self.weights, weighted_sum)  # 加权求和

        return weighted_sum
    
    def get_scaled_grid(self, shape, device, scale=1.0):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float, device=device)*scale # 直接指定设备
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx  # 不再需要再次移动到设备
    
# if __name__ == "__main__":
#     # 设置模型参数
#     modes = 500  # 选择合适的傅里叶模式数
#     width = 16  # 选择合适的宽度
#     num_subnets = 8  # 子网络数量
#     # 创建模型
#     model = MultiScaleFNO1d(modes, width)
#     # 打印模型信息
#     summary(model, input_size=(1, 1000, 1))  # 输入形状为 (batch_size, x=s, c=2)

################################################################
#  configurations
################################################################
ntrain = 800
ntest = 100
nval = ntest   # 验证集样本数与测试集相同  

sub = 10 #subsampling rate
h = 10000 // sub #total grid size divided by the subsampling rate
s = h + 1

batch_size = 25
learning_rate = 0.001
epochs = 500
iterations = epochs*(ntrain//batch_size)
 
modes = 500
width = 16

################################################################
# read data
################################################################

# 数据加载部分  
dataloader = MatReader(r"Helmholtz_data.mat")
x_data = dataloader.read_field('a_matrix')[:, ::sub][:,:s]
y_data = dataloader.read_field('u_matrix')[:, ::sub][:,:s]  

# 划分数据  
x_train = x_data[:ntrain, :]  
y_train = y_data[:ntrain, :]  

x_val = x_data[ntrain:ntrain + nval, :]  
y_val = y_data[ntrain:ntrain + nval, :]  

x_test = x_data[-ntest:, :]  
y_test = y_data[-ntest:, :]  

# reshape  
x_train = x_train.reshape(ntrain, s, 1)  
x_val = x_val.reshape(nval, s, 1)  
x_test = x_test.reshape(ntest, s, 1)  

# 创建 DataLoader  
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)


# 模型定义  
model = MultiScaleFNO1d(modes, width,num_subnets=8).cuda()  
print(f"Model parameters: {count_params(model)}")  

# 优化器与调度器  

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)  
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(epochs*0.8), eta_min=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# 损失函数  
myloss = LpLoss(size_average=False)  

# 初始化最佳验证损失为正无穷，以及对应的测试损失  
best_val_loss = float('inf')  
best_test_loss = float('inf')  # 用于记录验证损失最低时对应的测试损失  

# 路径设置  
best_model_path = 'saved_model.pt'
best_epoch = 1

log_file = 'saved_error.csv'
# 定义表头  
header = ['Epoch', 'Time(s)', 'Train L2', 'Val L2', 'Test L2']  

# 检查文件是否存在，如果不存在则创建并写入表头  
if not os.path.exists(log_file):  
    with open(log_file, mode='w', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(header)  
    print(f"日志文件 '{log_file}' 已创建并写入表头。")  
else:  
    print(f"日志文件 '{log_file}' 已存在，将在其后追加数据。")
    
# 训练与验证循环  
for ep in range(epochs):  
    # 训练阶段  
    model.train()  
    t1 = default_timer()  
    train_mse = 0  
    train_l2 = 0  
    for x_batch, y_batch in train_loader:  
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()  

        optimizer.zero_grad()  
        out = model(x_batch)  

        # mse = F.mse_loss(out.view(x_batch.size(0), -1), y_batch.view(y_batch.size(0), -1), reduction='mean')  
        l2 = myloss(out.view(x_batch.size(0), -1), y_batch.view(y_batch.size(0), -1))  
        l2.backward()  # 使用 L2 相对损失  

        optimizer.step()  
        scheduler.step()  
        # train_mse += mse.item() * x_batch.size(0)  
        train_l2 += l2.item()  

    # 计算训练集的平均损失  
    # train_mse /= ntrain  
    train_l2 /= ntrain  

    # 验证阶段  
    model.eval()  
    val_l2 = 0.0  
    with torch.no_grad():  
        for x_val_batch, y_val_batch in val_loader:  
            x_val_batch, y_val_batch = x_val_batch.cuda(), y_val_batch.cuda()  
            out_val = model(x_val_batch)  
            val_l2 += myloss(out_val.view(x_val_batch.size(0), -1), y_val_batch.view(y_val_batch.size(0), -1)).item()  

    val_l2 /= nval  

    # 测试阶段  
    test_l2 = 0.0  
    with torch.no_grad():  
        for x_test_batch, y_test_batch in test_loader:  
            x_test_batch, y_test_batch = x_test_batch.cuda(), y_test_batch.cuda()  
            out_test = model(x_test_batch)  
            test_l2 += myloss(out_test.view(x_test_batch.size(0), -1), y_test_batch.view(y_test_batch.size(0), -1)).item()  

    test_l2 /= ntest  

    t2 = default_timer()  

    print(f'Epoch: {ep+1}, Time: {t2 - t1:.2f}s,  Train L2: {train_l2:.6f}, Val L2: {val_l2:.6f}, Test L2: {test_l2:.6f}')  
    # 将数据写入日志文件  
    with open(log_file, mode='a', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow([  
            ep + 1,  
            f"{t2 - t1:.2f}",  
            f"{train_l2:.6f}",  
            f"{val_l2:.6f}",  
            f"{test_l2:.6f}"  
        ])  
    # 检查当前验证损失是否为最佳  
    if val_l2 < best_val_loss:  
        best_val_loss = val_l2  
        best_test_loss = test_l2  # 记录此时的测试损失 
        best_epoch = ep + 1 
        torch.save(model.state_dict(), best_model_path)  
        # print(f'验证集损失降低，模型参数已保存到 {best_model_path}，对应的测试集L2误差为 {best_test_loss:.6f}')  

# 输出验证误差最低时对应的测试误差  
print(f'\n训练完成。验证集损失最低时的测试集L2误差为: {best_test_loss:.6f}')  
print(f'最佳的Epoch数为: {best_epoch}')
# （可选）加载最佳模型并进行最终测试评估  
model.load_state_dict(torch.load(best_model_path))  
model.eval()  
final_test_l2 = 0.0  
with torch.no_grad():  
    for x_test_batch, y_test_batch in test_loader:  
        x_test_batch, y_test_batch = x_test_batch.cuda(), y_test_batch.cuda()  
        out_final = model(x_test_batch)  
        final_test_l2 += myloss(out_final.view(x_test_batch.size(0), -1), y_test_batch.view(y_test_batch.size(0), -1)).item()  

final_test_l2 /= ntest  

print(f'加载最佳模型后的测试集L2误差为: {final_test_l2:.6f}')
