"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from torchinfo import summary
import os
import csv  
import torch
import numpy as np
import random
# torch.cuda.set_device(2)  # 或1,2,3  
# CUDA_VISIBLE_DEVICES=2 taskset -c 0-39 python fourier_2d_earlystop_mscale.py  

torch.manual_seed(10)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)
    
    def sfm_activation(self, z, s=0.5):
        """ SFM 激活函数，输出是两个分量的和 """
        return torch.sin(z)  # 求和
    
    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.sfm_activation(x)

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
        # x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1).to(device)

class MultiScaleFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_subnets=8):
        super(MultiScaleFNO2d, self).__init__()

        self.num_subnets = num_subnets  
        self.subnets = nn.ModuleList([FNO2d(modes1, modes2, width) for _ in range(num_subnets)])  
        
        # Initialize scaling factors and weights  
        self.scaling_factors = nn.Parameter(torch.tensor([1.0, 40.0, 80.0, 100.0, 120.0, 140.0, 180.0, 200.0]))  # 可训练的缩放因子
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]))  
        # self.scaling_factors = nn.Parameter(torch.tensor([1.0, 4.0, 8.0, 10.0, 12.0, 14.0, 18.0, 20.0]))  
        self.weights = nn.Parameter(torch.tensor([1.0] * num_subnets))  # Equal weights initially  

    def forward(self, x):  
        outputs = []  
        
        for i, net in enumerate(self.subnets):  
            # Get scaled input while maintaining 2D structure  
            grid = self.get_scaled_grid(x.shape, x.device, scale=self.scaling_factors[i])  
            x_scaled = torch.cat((self.scaling_factors[i] * x, grid), dim=-1)  
            outputs.append(net(x_scaled))  
        
        # Weighted sum of outputs  
        weighted_sum = torch.stack(outputs, dim=0)  
        weighted_sum = torch.einsum('i,i...->...', self.weights, weighted_sum)  
        
        return weighted_sum  

    def get_scaled_grid(self, shape, device, scale=1.0):  
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]  
        
        # Scale both x and y dimensions  
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device) * scale  
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])  
        
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device) * scale  
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])  
        
        return torch.cat((gridx, gridy), dim=-1)  
################################################################  
# configs  
################################################################  
TRAIN_PATH = r"D:\Research\FNO\fourier_neural_operator-master\fourier_neural_operator-master\Helmholtz\data_2d\Helmholtz_2D_1200to400_w10_L50pi_U70pi_f11_50pi_70pi_lambda10_L01_10-3.mat"  

ntrain = 800 
ntest = 100  
nval = 100  # 验证集大小  
total_samples = ntrain + nval + ntest  # 总共需要的样本数  

batch_size = 25 
learning_rate = 0.001  
epochs = 500  
iterations = epochs*(ntrain//batch_size)  

modes = 100 
width = 10  

r =  2
h = int(((401 - 1)/r) + 1)  
s = h  

# 添加模型和日志保存路径  
best_model_path = 'mscale1_200_Helmholtz_2D_1200to400_w10_L50pi_U70pi_f11_50pi_70pi_lambda10_L01_10-3_500epoch_allsin.pt'
log_file = 'mscale1_200_Helmholtz_2D_1200to400_w10_L50pi_U70pi_f11_50pi_70pi_lambda10_L01_10-3_500epoch_allsin.csv'  

################################################################  
# load data and data normalization  
################################################################  
reader = MatReader(TRAIN_PATH)  
# 一次性读取所有需要的数据  
x_data = reader.read_field('a_matrix')[:total_samples,::r,::r][:,:s,:s]  
y_data = reader.read_field('u_matrix')[:total_samples,::r,::r][:,:s,:s]  

# 划分数据集  
x_train = x_data[:ntrain,: ,:]  
y_train = y_data[:ntrain,: ,:]  

x_val = x_data[ntrain:ntrain+nval,: ,:]  
y_val = y_data[ntrain:ntrain+nval,: ,:]  

x_test = x_data[ntrain+nval:total_samples,: ,:]  
y_test = y_data[ntrain+nval:total_samples,: ,:]  
 

# reshape  
x_train = x_train.reshape(ntrain,s,s,1)  
x_val = x_val.reshape(nval,s,s,1)  
x_test = x_test.reshape(ntest,s,s,1)  

# 创建数据加载器  
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)  
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)  
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)  

# 创建日志文件  
if not os.path.exists(log_file):  
    with open(log_file, mode='w', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(['Epoch', 'Time(s)', 'Train L2', 'Val L2', 'Test L2'])  
    print(f"日志文件 '{log_file}' 已创建并写入表头。")  
else:  
    print(f"日志文件 '{log_file}' 已存在，将在其后追加数据。")
  
################################################################  
# training and evaluation  
################################################################  
model = MultiScaleFNO2d(modes, modes, width).cuda()  
print(count_params(model))  

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)  

myloss = LpLoss(size_average=False)  

# 初始化最佳验证损失  
best_val_loss = float('inf')  
best_test_loss = float('inf')  
best_epoch = 1  

for ep in range(epochs):  
    # 训练阶段  
    model.train()  
    t1 = default_timer()  
    train_l2 = 0  
    
    for x, y in train_loader:  
        x, y = x.cuda(), y.cuda()  

        optimizer.zero_grad()  
        out = model(x).reshape(batch_size, s, s)  

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))  
        loss.backward()  

        optimizer.step()  
        scheduler.step()  
        train_l2 += loss.item()  

    # 验证阶段  
    model.eval()  
    val_l2 = 0.0  
    with torch.no_grad():  
        for x, y in val_loader:  
            x, y = x.cuda(), y.cuda()  
            out = model(x).reshape(batch_size, s, s)  
            val_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()  

    # 测试阶段  
    test_l2 = 0.0  
    with torch.no_grad():  
        for x, y in test_loader:  
            x, y = x.cuda(), y.cuda()  
            out = model(x).reshape(batch_size, s, s)  
            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()  

    train_l2 /= ntrain  
    val_l2 /= nval  
    test_l2 /= ntest  

    t2 = default_timer()  
    
    # 记录日志  
    with open(log_file, mode='a', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow([ep + 1, f"{t2-t1:.2f}", f"{train_l2:.6f}", f"{val_l2:.6f}", f"{test_l2:.6f}"])  

    # 检查是否为最佳模型，保存模型和归一化器  
    if val_l2 < best_val_loss:  
        best_val_loss = val_l2  
        best_test_loss = test_l2  
        best_epoch = ep + 1  
        
        # 保存模型和归一化器  
        torch.save({  
            'model_state_dict': model.state_dict(),  
            'epoch': ep + 1,  
            'best_val_loss': best_val_loss,  
            'best_test_loss': best_test_loss,  
            'config': {  
                'modes': modes,  
                'width': width,  
                'r': r,  
                's': s  
            }  
        }, best_model_path)  

    # print(ep, t2-t1, train_l2, val_l2, test_l2)  

# 输出最终结果  
print(f'\n训练完成。验证集损失最低时的测试集L2误差为: {best_test_loss:.6f}')  
print(f'最佳的Epoch数为: {best_epoch}')  

# # 加载最佳模型进行最终测试  
# checkpoint = torch.load('/zilin/results_1_14/mscale_model_Helmholtz_2Dcom_Orthogonal_w50_norm_-11_rand_L0_U20pi_f10_L90_U100_fixed_lambda2_c0.9_res200_-1_1_1layer.pt')  
# model.load_state_dict(checkpoint['model_state_dict'])  
# x_normalizer = checkpoint['x_normalizer']  
# y_normalizer = checkpoint['y_normalizer']  

# model.eval()  
# final_test_l2 = 0.0  
# with torch.no_grad():  
#     for x, y in test_loader:  
#         x, y = x.cuda(), y.cuda()  
#         out = model(x).reshape(batch_size, s, s)  
#         out = y_normalizer.decode(out)
#         y = y_normalizer.decode(y)    
#         final_test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()  

# final_test_l2 /= ntest  
# print(f'加载最佳模型后的测试集L2误差为: {final_test_l2:.6f}')  

# ################################################################  
# # prediction function  
# ################################################################  
# def predict(model, x_normalizer, y_normalizer, x):  
#     """  
#     使用保存的模型和归一化器进行预测  
    
#     Args:  
#         model: 加载的模型  
#         x_normalizer: 输入归一化器  
#         y_normalizer: 输出归一化器  
#         x: 输入数据  
    
#     Returns:  
#         预测结果  
#     """  
#     model.eval()  
#     with torch.no_grad():  
#         x = x_normalizer.encode(x)  
#         x = x.reshape(-1, s, s, 1)  
#         x = x.cuda()  
#         out = model(x).reshape(-1, s, s)  
#         out = y_normalizer.decode(out)  
#     return out  

# def load_and_predict(x_new):  
#     """  
#     加载模型和归一化器并进行预测  
    
#     Args:  
#         x_new: 新的输入数据  
    
#     Returns:  
#         预测结果  
#     """  
#     # 加载模型和归一化器  
#     checkpoint = torch.load(best_model_path)  
#     model.load_state_dict(checkpoint['model_state_dict'])  
#     x_normalizer = checkpoint['x_normalizer']  
#     y_normalizer = checkpoint['y_normalizer']  
    
#     # 预测  
#     return predict(model, x_normalizer, y_normalizer, x_new)