import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


"""
VAE_Encoder
VAE_Encoder 类是变分自编码器（VAE）的编码器部分，负责将输入图像数据编码为潜在空间表示（latent representation）。
该编码器通过一系列卷积层、残差块和自注意力机制，逐步降低输入图像的空间分辨率，同时增加通道数，
最终输出均值（mean）和对数方差（log variance），用于生成潜在空间的随机样本。
VAE_Encoder 类通过一系列的卷积层、残差块和自注意力机制，将输入图像数据编码为潜在空间的均值和对数方差。
具体步骤如下：
1.输入图像处理: 输入图像首先通过3x3卷积层，转换为128个通道。
2.下采样: 通过步幅为2的卷积层逐步降低图像的空间分辨率，同时增加通道数。
3.残差连接: 在每个分辨率阶段，使用残差块（Residual Block）

一 初始化参数: 
二 主要组件:
卷积层:
nn.Conv2d(3, 128, kernel_size=3, padding=1): 3x3卷积层，将输入图像从3个通道（RGB）转换为128个通道，输出尺寸与输入相同。
nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0): 3x3卷积层，步幅为2，将特征图的空间尺寸减半（高度和宽度各减半），通道数保持128。
nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0): 3x3卷积层，步幅为2，将特征图的空间尺寸再次减半，通道数从128增加到256。
nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0): 3x3卷积层，步幅为2，将特征图的空间尺寸再次减半，通道数从256增加到512。
nn.Conv2d(512, 8, kernel_size=3, padding=1): 3x3卷积层，将通道数从512减少到8，输出尺寸与输入相同。
nn.Conv2d(8, 8, kernel_size=1, padding=0): 1x1卷积层，保持通道数不变，用于调整特征图的维度。

残差块:
VAE_ResidualBlock(128, 128): 包含两个3x3卷积层和组归一化层，通道数保持为128。
VAE_ResidualBlock(128, 128): 同上。
VAE_ResidualBlock(256, 256): 通道数保持为256。
VAE_ResidualBlock(256, 512): 通道数从256增加到512。
VAE_ResidualBlock(512, 512): 通道数保持为512。
VAE_ResidualBlock(512, 512): 同上。
VAE_ResidualBlock(512, 512): 同上。
VAE_ResidualBlock(512, 512): 同上。

自注意力块:
VAE_AttentionBlock(512): 使用自注意力机制处理通道数为512的特征图。

归一化和激活函数:
nn.GroupNorm(32, 512): 组归一化层，组数为32，通道数为512。
nn.SiLU(): SiLU激活函数。


详细步骤：
1.初始化:
定义一系列卷积层、残差块、自注意力块和归一化层，按顺序添加到 nn.Sequential 中。

2.前向传播方法 (forward):
输入:
x: 输入图像，形状为 (Batch_Size, Channel, Height, Width)。
noise: 随机噪声，形状为 (Batch_Size, 4, Height / 8, Width / 8)。

不对称填充:
对于具有步幅为2的卷积层（即下采样操作），对输入 x 进行不对称填充，在右侧和底部填充1个像素，以确保输出尺寸与预期一致。
填充后，x 的形状从 (Batch_Size, Channel, Height, Width) 变为 (Batch_Size, Channel, Height + 1, Width + 1)。

层处理:
将 x 依次通过所有定义好的层进行处理，包括卷积层、残差块和自注意力块。

均值和对数方差计算:
将输出 x 拆分为两个张量，均值（mean）和对数方差（log_variance），每个张量的形状为 (Batch_Size, 4, Height / 8, Width / 8)。

方差计算:
对对数方差进行裁剪，限制在-30到20之间，以确保方差的范围在约1e-14到1e8之间。
计算方差（variance）为对数方差的指数。
计算标准差（stdev）为方差的平方根。

潜在空间样本生成:
生成潜在空间的样本，通过将均值与标准差的乘积与噪声相加，形状保持为 (Batch_Size, 4, Height / 8, Width / 8)。

缩放调整:
将生成的样本乘以常数0.18215，以调整尺度。

输出:
返回调整后的潜在空间样本，形状为 (Batch_Size, 4, Height / 8, Width / 8)。

"""

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(128, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(256, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            # 因为 padding=1，意味着宽度和高度会增加2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # 由于 padding=1 表示 Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1
            # 因此 Out_Width = In_Width + 2（高度同理），这将补偿3x3卷积核的影响
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)
        # 输入 x 的形状: (Batch_Size, Channel, Height, Width)
        # 输入 noise 的形状: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # 在下采样时应该进行不对称填充（参见 #8）
                # 填充: (左, 右, 上, 下)
                # 在右侧和底部填充0
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # 将对数方差限制在-30到20之间，以便方差在（大约）1e-14到1e8之间
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # 将 N(0, 1) 转换为 N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        # 乘以一个常数进行缩放
        x *= 0.18215
        
        return x

