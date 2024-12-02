import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention



"""
VAE_AttentionBlock
VAE_AttentionBlock 类结合了组归一化（Group Normalization）和自注意力机制（Self-Attention），
用于处理输入特征图。它通过组归一化进行特征标准化，然后应用自注意力机制捕捉特征图内部的空间依赖关系。

一 初始化参数: channels 
二 主要组件: self.groupnorm, self.attention

详细步骤：
1.初始化:
初始化组归一化层，设定组数为32，通道数为输入参数 channels。
初始化自注意力层，使用1个注意力头，嵌入维度为 channels。

2.前向传播方法 (forward):
输入: x 的形状为 (Batch_Size, Features, Height, Width)。
残差连接: 保存输入 x 作为残差连接的一部分。
组归一化: 对输入 x 应用组归一化，形状保持不变。
调整形状: 将 x 的形状从 (Batch_Size, Features, Height, Width) 调整为 (Batch_Size, Features, Height * Width)，以便进行自注意力计算。
转置: 将 x 的形状从 (Batch_Size, Features, Height * Width) 转换为 (Batch_Size, Height * Width, Features)。每个像素点被视为一个特征，序列长度为 Height * Width。
自注意力计算:对调整后的 x 应用自注意力机制，不使用因果掩码。输出 x 的形状为 (Batch_Size, Height * Width, Features)。
恢复形状: 将 x 的形状从 (Batch_Size, Height * Width, Features) 转置回 (Batch_Size, Features, Height * Width)。
恢复原始形状: 将 x 的形状从 (Batch_Size, Features, Height * Width) 恢复为 (Batch_Size, Features, Height, Width)。
残差连接: 将处理后的 x 与之前保存的残差连接部分相加。
输出: 返回形状为 (Batch_Size, Features, Height, Width) 的最终结果。

"""

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 初始化组归一化层，组数为32，通道数为输入参数 channels
        self.groupnorm = nn.GroupNorm(32, channels)
         # 初始化自注意力层，使用1个注意力头，嵌入维度为 channels
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        # x: (Batch_Size, Features, Height, Width)
        # 输入 x 的形状: (Batch_Size, Features, Height, Width)

        # 保存输入 x 作为残差连接的一部分
        residue = x 

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        # 对输入 x 应用组归一化，形状保持不变
        x = self.groupnorm(x)

        # 获取输入 x 的维度信息
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        # 每个像素点被视为一个特征，序列长度为 Height * Width
        x = x.transpose(-1, -2)
        
        # Perform self-attention WITHOUT mask
        # 对调整后的 x 应用自注意力机制，不使用因果掩码
        # 输出 x 的形状为 (Batch_Size, Height * Width, Features)
        x = self.attention(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))
        
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
        # 将处理后的 x 与之前保存的残差连接部分相加
        x += residue

        # (Batch_Size, Features, Height, Width)
        return x 
    


"""
VAE_ResidualBlock
VAE_ResidualBlock 类实现了残差连接（Residual Connection）的卷积神经网络块。
它通过两个卷积层和组归一化层来处理输入数据，并使用残差连接来缓解深层网络中的梯度消失问题。

一 初始化参数: in_channels, out_channels 
二 主要组件: self.groupnorm_1, self.conv_1, self.groupnorm_2, self.conv_2, self.residual_layer

详细步骤：
1.初始化:
初始化第一个组归一化层，组数为32，通道数为 in_channels。
初始化第一个卷积层，输入通道数为 in_channels，输出通道数为 out_channels，卷积核大小为3，填充为1。
初始化第二个组归一化层，组数为32，通道数为 out_channels。
初始化第二个卷积层，输入和输出通道数均为 out_channels，卷积核大小为3，填充为1。
根据 in_channels 和 out_channels 是否相等，初始化残差连接层：
如果相等，使用恒等映射。
如果不相等，使用1x1卷积层进行通道数匹配。

2.前向传播方法 (forward):
输入: x 的形状为 (Batch_Size, In_Channels, Height, Width)。
残差连接: 保存输入 x 作为残差连接的一部分。
第一个组归一化: 对输入 x 应用第一个组归一化，形状保持不变。
激活函数: 对归一化后的 x 应用 SiLU 激活函数。
第一个卷积: 对激活后的 x 应用第一个卷积层，将通道数从 in_channels 转换为 out_channels，形状保持不变。
第二个组归一化: 对第一个卷积层的输出应用第二个组归一化，形状保持不变。
激活函数: 对归一化后的输出应用 SiLU 激活函数。
第二个卷积: 对激活后的输出应用第二个卷积层，保持输出通道数为 out_channels，形状保持不变。
残差连接: 将处理后的输出与经过残差连接层调整后的输入相加，得到最终输出。

"""
  
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 初始化第一个组归一化层，组数为32，通道数为 in_channels
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # 初始化第一个卷积层，输入通道数为 in_channels，输出通道数为 out_channels，卷积核大小为3，填充为1
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 初始化第二个组归一化层，组数为32，通道数为 out_channels
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # 初始化第二个卷积层，输入和输出通道数均为 out_channels，卷积核大小为3，填充为1
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 如果输入通道数与输出通道数相等，则使用恒等映射作为残差连接层
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # 否则，使用1x1卷积层进行通道数匹配
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)
        # 输入 x 的形状: (Batch_Size, In_Channels, Height, Width)

        # 保存输入 x 作为残差连接的一部分
        residue = x

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        # 对输入 x 应用第一个组归一化，形状保持不变
        x = self.groupnorm_1(x)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        # 对归一化后的 x 应用 SiLU 激活函数
        x = F.silu(x)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对激活后的 x 应用第一个卷积层，将通道数从 in_channels 转换为 out_channels，形状保持不变
        x = self.conv_1(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对第一个卷积层的输出应用第二个组归一化，形状保持不变
        x = self.groupnorm_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对归一化后的输出应用 SiLU 激活函数
        x = F.silu(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对激活后的输出应用第二个卷积层，保持输出通道数为 out_channels，形状保持不变
        x = self.conv_2(x)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 将处理后的输出与经过残差连接层调整后的输入相加，得到最终输出
        return x + self.residual_layer(residue)



"""
VAE_Decoder
VAE_Decoder 类是一个用于变分自编码器（VAE）解码阶段的神经网络模块。
它通过一系列卷积层、残差块、自注意力块和上采样层，逐步将输入的低分辨率特征图解码为高分辨率的图像输出。
VAE_Decoder 类通过一系列的卷积层、残差块和上采样层，逐步将低分辨率的输入特征图解码为高分辨率的图像输出。
解码过程包括以下几个步骤：
1.输入调整: 对输入进行缩放调整，移除编码器中引入的缩放因子。
2.特征提取: 通过多个残差块和卷积层提取特征。
3.注意力机制: 应用自注意力块增强特征表示。
4.上采样: 通过上采样层逐步增加特征图的空间尺寸。
5.输出生成: 最终通过卷积层生成RGB图像输出。
这种结构使得模型能够有效地从潜在空间重建高质量的图像。

一 初始化参数: in_channels, out_channels 
二 主要组件:
卷积层:
nn.Conv2d(4, 4, kernel_size=1, padding=0): 1x1卷积层，保持通道数不变。
nn.Conv2d(4, 512, kernel_size=3, padding=1): 3x3卷积层，将通道数从4增加到512。
nn.Conv2d(512, 512, kernel_size=3, padding=1): 3x3卷积层，保持通道数为512。
nn.Conv2d(512, 256, kernel_size=3, padding=1): 3x3卷积层，将通道数从512减少到256。
nn.Conv2d(256, 128, kernel_size=3, padding=1): 3x3卷积层，将通道数从256减少到128。
nn.Conv2d(128, 3, kernel_size=3, padding=1): 3x3卷积层，将通道数从128减少到3，用于生成RGB图像。

残差块:
VAE_ResidualBlock(512, 512): 包含两个3x3卷积层和组归一化层，通道数保持为512。
VAE_ResidualBlock(512, 512): 同上。
VAE_ResidualBlock(512, 512): 同上。
VAE_ResidualBlock(512, 512): 同上。
VAE_ResidualBlock(512, 256): 通道数从512减少到256。
VAE_ResidualBlock(256, 256): 通道数保持为256。
VAE_ResidualBlock(256, 256): 同上。
VAE_ResidualBlock(128, 128): 通道数保持为128。

自注意力块:
VAE_AttentionBlock(512): 使用自注意力机制处理通道数为512的特征图。

上采样层:
nn.Upsample(scale_factor=2): 通过双线性插值将特征图的空间尺寸放大2倍。

归一化和激活函数:
nn.GroupNorm(32, 128): 组归一化层，组数为32，通道数为128。
nn.SiLU(): SiLU激活函数。


详细步骤：
1.初始化:
定义一系列卷积层、残差块、自注意力块、上采样层和归一化层，按顺序添加到 nn.Sequential 中。

2.前向传播方法 (forward):
输入: x 的形状为 (Batch_Size, 4, Height / 8, Width / 8)。
缩放调整: 将输入 x 除以0.18215，以移除编码器中引入的缩放因子。
层处理: 将 x 依次通过所有定义好的层进行处理。
输出: 返回处理后的 x，形状为 (Batch_Size, 3, Height, Width)，对应于RGB图像。

"""

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.SiLU(), 
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # 输入 x 的形状: (Batch_Size, 4, Height / 8, Width / 8)
        
        # Remove the scaling added by the Encoder.
        # 移除编码器中引入的缩放因子
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x
      
