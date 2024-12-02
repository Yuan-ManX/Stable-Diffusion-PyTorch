import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention



"""
TimeEmbedding
TimeEmbedding 类用于将时间步信息嵌入到潜在空间中，常用于处理序列数据或时间序列数据。
该类通过两个线性层和一个激活函数，将输入的时间步信息进行非线性变换，从而生成一个更高维度的嵌入表示。

TimeEmbedding 类通过两个线性层和一个 SiLU 激活函数，将输入的时间步信息进行非线性变换，生成一个更高维度的嵌入表示。
具体步骤如下：
1.线性变换: 输入向量通过第一个线性层，从 n_embd 维度增加到 4 * n_embd。
2.激活函数: 应用 SiLU 激活函数，增加非线性特性。
3.线性变换: 再次通过第二个线性层，保持输出维度为 4 * n_embd。
这种结构能够有效地将时间步信息嵌入到更高维度的空间中，适用于处理序列数据或时间序列数据的任务。

一 初始化参数: n_embd 
二 主要组件: self.linear_1, self.linear_2

详细步骤：
1.初始化:
初始化第一个线性层，将输入维度从 n_embd 增加到 4 * n_embd。
初始化第二个线性层，保持输出维度为 4 * n_embd。

2.前向传播方法 (forward):
输入: x 的形状为 (1, 320)，表示一个时间步的嵌入向量。
线性变换:将输入 x 通过第一个线性层 self.linear_1，从 (1, 320) 转换为 (1, 1280)。
激活函数:对变换后的 x 应用 SiLU 激活函数，保持形状为 (1, 1280)。
线性变换:将激活后的 x 通过第二个线性层 self.linear_2，保持形状为 (1, 1280)。
输出: 返回最终的嵌入向量 x，形状为 (1, 1280)。

"""

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # 初始化第一个线性层，将输入维度从 n_embd 增加到 4 * n_embd
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # 初始化第二个线性层，保持输出维度为 4 * n_embd
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)
        # 输入 x 的形状: (1, 320)

        # (1, 320) -> (1, 1280)
        # 通过第一个线性层，将 (1, 320) 转换为 (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        # 对变换后的 x 应用 SiLU 激活函数，保持形状为 (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        # 通过第二个线性层，保持形状为 (1, 1280)
        x = self.linear_2(x)

        return x



"""
UNET_ResidualBlock
UNET_ResidualBlock 类实现了 UNet 架构中的残差连接块，用于图像处理任务。
它结合了特征图和时间嵌入信息，通过一系列的组归一化、卷积层和线性变换，增强了模型的表达能力，并利用残差连接缓解梯度消失问题。

UNET_ResidualBlock 类通过特征图和时间嵌入信息的结合，利用组归一化、卷积层和线性变换，增强了模型的表达能力。
具体步骤如下：
1.特征图处理:输入特征图通过组归一化和卷积层进行变换。应用 SiLU 激活函数增加非线性。
2.时间嵌入处理:时间嵌入向量通过线性层转换为与输出通道数相同的维度。应用 SiLU 激活函数。
3.合并与处理:将时间嵌入信息与特征图合并。对合并后的特征图进行组归一化和卷积层处理。应用 SiLU 激活函数。
4.残差连接:将处理后的特征图与原始输入特征图相加，实现残差连接，缓解梯度消失问题。
这种结构在图像处理任务中能够有效捕捉时间信息和空间信息，提升模型的性能。

一 初始化参数:
in_channels (int): 输入特征的通道数。
out_channels (int): 输出特征的通道数。
n_time (int): 时间嵌入的维度，默认为1280。

二 主要组件:
特征处理:
self.groupnorm_feature: 组归一化层，用于对输入特征图进行归一化，组数为32。
self.conv_feature: 卷积层，将输入特征图的通道数从 in_channels 转换为 out_channels，卷积核大小为3，填充为1。

时间嵌入处理:
self.linear_time: 线性层，将时间嵌入从 n_time 维度转换为 out_channels 维度。

合并与处理:
self.groupnorm_merged: 组归一化层，用于对合并后的特征图进行归一化，组数为32。
self.conv_merged: 卷积层，保持输出通道数为 out_channels，卷积核大小为3，填充为1。

残差连接:
self.residual_layer: 残差连接层。如果 in_channels 与 out_channels 相等，使用恒等映射；否则，使用1x1卷积层进行通道数匹配。


详细步骤：
1.初始化:
初始化特征处理的组归一化层和卷积层。
初始化时间嵌入的线性层。
初始化合并后的组归一化层和卷积层。
根据 in_channels 和 out_channels 是否相等，初始化残差连接层：如果相等，使用恒等映射。如果不相等，使用1x1卷积层进行通道数匹配。

2.前向传播方法 (forward):
输入:
feature: 输入特征图，形状为 (Batch_Size, In_Channels, Height, Width)。
time: 时间嵌入向量，形状为 (1, 1280)。

残差连接: 保存输入 feature 作为残差连接的一部分。

特征处理:
对 feature 应用组归一化，保持形状不变。
对归一化后的 feature 应用 SiLU 激活函数。
对激活后的 feature 应用卷积层，将通道数从 in_channels 转换为 out_channels，形状保持不变。

时间嵌入处理:
对 time 应用 SiLU 激活函数，保持形状为 (1, 1280)。
通过线性层将 time 转换为 (1, Out_Channels)。

合并特征和时间信息:
将 time 向量扩展为 (1, Out_Channels, 1, 1)，并与 feature 相加，得到合并后的特征图 merged。

合并后的处理:
对 merged 应用组归一化，保持形状不变。
对归一化后的 merged 应用 SiLU 激活函数。
对激活后的 merged 应用卷积层，保持输出通道数为 out_channels，形状保持不变。

残差连接: 将处理后的 merged 与经过残差连接层调整后的 residue 相加，得到最终输出。

"""

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        # 初始化组归一化层，用于对输入特征图进行归一化，组数为32
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        # 初始化卷积层，将输入特征图的通道数从 in_channels 转换为 out_channels，卷积核大小为3，填充为1
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 初始化线性层，将时间嵌入从 n_time 维度转换为 out_channels 维度
        self.linear_time = nn.Linear(n_time, out_channels)

        # 初始化合并后的组归一化层，组数为32
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        # 初始化卷积层，保持输出通道数为 out_channels，卷积核大小为3，填充为1
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 如果输入通道数与输出通道数相等，则使用恒等映射作为残差连接层
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # 否则，使用1x1卷积层进行通道数匹配
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)
        # 输入 feature 的形状: (Batch_Size, In_Channels, Height, Width)
        # 输入 time 的形状: (1, 1280)

        # 保存输入 feature 作为残差连接的一部分
        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        # 对 feature 应用组归一化，保持形状不变
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        # 对归一化后的 feature 应用 SiLU 激活函数
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对激活后的 feature 应用卷积层，将通道数从 in_channels 转换为 out_channels，形状保持不变
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        # 对 time 应用 SiLU 激活函数，保持形状为 (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        # 通过线性层将 time 转换为 (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        # 将 time 向量扩展为 (1, Out_Channels, 1, 1)，并与 feature 相加，得到合并后的特征图 merged
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对 merged 应用组归一化，保持形状不变
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对归一化后的 merged 应用 SiLU 激活函数
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 对激活后的 merged 应用卷积层，保持输出通道数为 out_channels，形状保持不变
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # 将处理后的 merged 与经过残差连接层调整后的 residue 相加，得到最终输出
        return merged + self.residual_layer(residue)



"""
UNET_AttentionBlock
UNET_AttentionBlock 类是一个用于 UNet 架构的自注意力模块，旨在捕捉输入特征图的空间依赖关系。
它结合了自注意力（Self-Attention）和交叉注意力（Cross-Attention）机制，
以及前馈网络（FFN）和跳跃连接（skip connections），以增强特征表示能力。

一 初始化参数:
n_head (int): 注意力头的数量，用于多头注意力机制。
n_embd (int): 每个注意力头的嵌入维度。
d_context (int): 交叉注意力的上下文嵌入维度，默认为768。

二 主要组件:

归一化和卷积层:
self.groupnorm: 组归一化层，用于对输入特征图进行归一化，组数为32，通道数为 n_head * n_embd。
self.conv_input: 1x1卷积层，保持输入特征图的通道数不变。
self.conv_output: 1x1卷积层，用于调整最终输出的通道数。

注意力机制:
self.layernorm_1: 第一个层归一化层，用于自注意力前的归一化。
self.attention_1: 自注意力层，使用 SelfAttention 类实现，不使用投影偏置。
self.layernorm_2: 第二个层归一化层，用于交叉注意力前的归一化。
self.attention_2: 交叉注意力层，使用 CrossAttention 类实现，包含上下文嵌入维度 d_context，不使用投影偏置。

前馈网络 (FFN) 和 GeGLU 激活:
self.layernorm_3: 第三个层归一化层，用于 FFN 前的归一化。
self.linear_geglu_1: 线性层，将输入维度扩展到 4 * channels * 2，用于 GeGLU 激活。
self.linear_geglu_2: 线性层，将 GeGLU 激活后的维度缩减回 channels。


详细步骤:

1.初始化:
初始化组归一化层和1x1卷积层。
初始化层归一化层和自注意力层。
初始化交叉注意力层。
初始化前馈网络的线性层和 GeGLU 激活层。

2.前向传播方法 (forward):
输入:
x: 输入特征图，形状为 (Batch_Size, Features, Height, Width)。
context: 上下文嵌入，形状为 (Batch_Size, Seq_Len, Dim)。

残差连接: 保存输入 x 作为长残差连接的一部分。

组归一化和卷积:
对 x 应用组归一化，保持形状不变。
通过1x1卷积层调整 x 的通道数，形状保持不变。

调整形状:
将 x 的形状从 (Batch_Size, Features, Height, Width) 调整为 (Batch_Size, Features, Height * Width)。
转置 x 的形状为 (Batch_Size, Height * Width, Features)，以便进行注意力计算。

自注意力:
保存 x 作为短残差连接的一部分。
对 x 应用层归一化。
应用自注意力层，输出形状保持为 (Batch_Size, Height * Width, Features)。
将自注意力输出与短残差连接相加。

交叉注意力:
保存 x 作为新的短残差连接。
对 x 应用层归一化。
应用交叉注意力层，输入上下文嵌入 context，输出形状保持为 (Batch_Size, Height * Width, Features)。
将交叉注意力输出与短残差连接相加。

前馈网络 (FFN) 和 GeGLU 激活:
保存 x 作为新的短残差连接。
对 x 应用层归一化。
通过线性层和 GeGLU 激活函数处理 x，将维度扩展到 (Batch_Size, Height * Width, Features * 4)。
应用 GeGLU 激活函数，将 x 与门控信号相乘。
通过线性层将维度缩减回 (Batch_Size, Height * Width, Features)。
将 FFN 输出与短残差连接相加。

恢复形状:
将 x 的形状转置回 (Batch_Size, Features, Height * Width)。
将 x 的形状恢复为 (Batch_Size, Features, Height, Width)。

最终跳跃连接:
将处理后的 x 与初始输入的残差连接部分相加，通过1x1卷积层调整通道数，得到最终输出。

"""

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        # 初始化组归一化层，组数为32，通道数为 channels，epsilon 为1e-6
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        # 初始化1x1卷积层，保持输入特征图的通道数不变
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # 初始化层归一化层，用于自注意力前的归一化
        self.layernorm_1 = nn.LayerNorm(channels)
        # 初始化自注意力层，不使用投影偏置
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        # 初始化层归一化层，用于交叉注意力前的归一化
        self.layernorm_2 = nn.LayerNorm(channels)
        # 初始化交叉注意力层，包含上下文嵌入维度 d_context，不使用投影偏置
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        # 初始化层归一化层，用于 FFN 前的归一化
        self.layernorm_3 = nn.LayerNorm(channels)
        # 初始化线性层，将输入维度扩展到 4 * channels * 2，用于 GeGLU 激活
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        # 初始化线性层，将 GeGLU 激活后的维度缩减回 channels
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # 初始化1x1卷积层，用于调整最终输出的通道数
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        # 输入 x 的形状: (Batch_Size, Features, Height, Width)
        # 输入 context 的形状: (Batch_Size, Seq_Len, Dim)

        # 保存输入 x 作为长残差连接的一部分
        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        # 对 x 应用组归一化，保持形状不变
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        # 通过1x1卷积层调整 x 的通道数，形状保持不变
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        # 将 x 的形状转置为 (Batch_Size, Height * Width, Features)，以便进行注意力计算
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        # 保存 x 作为短残差连接的一部分
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 对 x 应用层归一化
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 应用自注意力层，输出形状保持为 (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 将自注意力输出与短残差连接相加
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        # 保存 x 作为新的短残差连接
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 对 x 应用层归一化
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 应用交叉注意力层，输入上下文嵌入 context，输出形状保持为 (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 将交叉注意力输出与短残差连接相加
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        # 保存 x 作为新的短残差连接
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 对 x 应用层归一化
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
         # 通过线性层和 GeGLU 激活函数处理 x，将维度扩展到 (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        # 应用 GeGLU 激活函数，将 x 与门控信号相乘
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        # 通过线性层将维度缩减回 (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        # 将 FFN 输出与短残差连接相加
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        # 将 x 的形状转置回 (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        # 将 x 的形状恢复为 (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        # 最终跳跃连接，将处理后的 x 与初始输入的残差连接结合
        return self.conv_output(x) + residue_long



"""
Upsample
Upsample 类用于在图像处理任务中放大特征图的空间尺寸。
它通过最近邻插值（nearest neighbor interpolation）将特征图的空间尺寸放大两倍，
然后通过一个卷积层对插值后的特征图进行平滑处理，以减少插值可能带来的伪影。

Upsample 类通过以下步骤实现特征图的空间尺寸放大：
1.最近邻插值: 使用最近邻插值将输入特征图的空间尺寸放大两倍。
2.卷积平滑: 对插值后的特征图应用卷积层进行平滑处理，减少插值可能带来的伪影。
这种上采样方法在图像生成和图像分割等任务中广泛应用，能够有效地增加特征图的空间分辨率，同时保持特征图的通道数不变。

一 初始化参数:
channels (int): 输入特征图的通道数。

二 主要组件:
self.conv: 卷积层，卷积核大小为3，填充为1，用于对插值后的特征图进行卷积操作。


详细步骤:
1.初始化:
初始化一个卷积层，输入和输出通道数均为 channels，卷积核大小为3，填充为1。

2.前向传播方法 (forward):
输入: x 的形状为 (Batch_Size, Features, Height, Width)。

上采样:使用最近邻插值将特征图 x 的空间尺寸放大两倍，输出形状为 (Batch_Size, Features, Height * 2, Width * 2)。

卷积平滑:对上采样后的特征图应用卷积层 self.conv，进行平滑处理，输出形状保持为 (Batch_Size, Features, Height * 2, Width * 2)。

输出: 返回处理后的特征图。

"""

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 初始化卷积层，输入和输出通道数均为 channels，卷积核大小为3，填充为1
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        # 输入 x 的形状: (Batch_Size, Features, Height, Width)

        # 使用最近邻插值将特征图的空间尺寸放大两倍，输出形状为 (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        # 对上采样后的特征图应用卷积层，进行平滑处理
        return self.conv(x)



"""
SwitchSequential
SwitchSequential 类继承自 nn.Sequential，用于在模型的前向传播过程中，根据不同层的类型选择性地传递不同的参数。
它能够根据层的具体类型（如 UNET_AttentionBlock 或 UNET_ResidualBlock），
动态地将不同的输入参数传递给相应的层，从而实现更灵活和动态的模型结构。

一 初始化参数:
二 主要组件:


详细步骤:
1.初始化:
初始化时，通过继承 nn.Sequential，将一系列层按顺序添加到 SwitchSequential 中。

2.前向传播方法 (forward):
输入:
x: 输入张量，形状根据具体任务而定。
context: 上下文嵌入，形状为 (Batch_Size, Seq_Len, Dim)。
time: 时间步嵌入，形状为 (1, 1280)。

层处理:遍历 SwitchSequential 中的所有层。
对于每一层，检查其类型：
如果是 UNET_AttentionBlock 类型，则调用该层的 forward 方法，传入参数 x 和 context。
如果是 UNET_ResidualBlock 类型，则调用该层的 forward 方法，传入参数 x 和 time。
对于其他类型的层，直接调用该层的 forward 方法，传入参数 x。

输出: 返回处理后的最终输出张量。

"""

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        # 遍历所有层
        for layer in self:
            # 如果层是 UNET_AttentionBlock 类型，则传入 x 和 context
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            # 如果层是 UNET_ResidualBlock 类型，则传入 x 和 time
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                # 对于其他类型的层，直接传入 x
                x = layer(x)
        return x



"""
UNET
UNET 类实现了 UNet 架构，该架构通过编码器（encoder）和解码器（decoder）结构，结合跳跃连接（skip connections），在捕捉图像特征的同时保留空间信息。

一 初始化参数:
二 主要组件:
编码器（Encoders）:
由多个 SwitchSequential 模块组成，每个模块包含一个或多个层（如卷积层、残差块和注意力块）。
每个 SwitchSequential 模块逐步将输入图像的空间尺寸减半，同时增加通道数。
具体层配置如下：
第一层: 3x3卷积层，将通道数从4增加到320。
第二层: 残差块（320通道）和注意力块（8头，每头40维）。
第三层: 重复第二层的配置。
下采样: 3x3卷积层，步幅为2，将空间尺寸减半。
第四层: 残差块（320通道扩展到640）和注意力块（8头，每头80维）。
第五层: 重复第四层的配置。
下采样: 3x3卷积层，步幅为2，将空间尺寸再次减半。
第六层: 残差块（640通道扩展到1280）和注意力块（8头，每头160维）。
第七层: 重复第六层的配置。
下采样: 3x3卷积层，步幅为2，将空间尺寸再次减半。
第八层: 残差块（1280通道）和注意力块（可选）。

瓶颈层（Bottleneck）:
由多个层组成，进一步处理编码器输出的特征图。
具体层配置：
残差块（1280通道）。
注意力块（8头，每头160维）。
残差块（1280通道）。

解码器（Decoders）:
由多个 SwitchSequential 模块组成，每个模块包含一个或多个层（如残差块、注意力块和上采样层）。
每个 SwitchSequential 模块逐步将特征图的空间尺寸放大，同时减少通道数。
具体层配置如下：
第一层: 残差块（2560通道缩减到1280）。
第二层: 重复第一层的配置。
上采样: 上采样层，将空间尺寸放大两倍。
第三层: 残差块（2560通道缩减到1280）和注意力块（8头，每头160维）。
第四层: 重复第三层的配置。
上采样: 上采样层，将空间尺寸放大两倍。
第五层: 残差块（1920通道缩减到1280）和注意力块（8头，每头160维）。
第六层: 重复第五层的配置。
上采样: 上采样层，将空间尺寸放大两倍。
第七层: 残差块（960通道缩减到640）和注意力块（8头，每头80维）。
第八层: 重复第七层的配置。
上采样: 上采样层，将空间尺寸放大两倍。
第九层: 残差块（960通道缩减到320）和注意力块（8头，每头40维）。
第十层: 重复第九层的配置。

跳跃连接（Skip Connections）:
在编码器和解码器之间，通过跳跃连接将编码器各层的输出保存，并在解码器相应层中与解码器的输出进行拼接（concatenation），以保留空间信息。


详细步骤:
1.初始化:
初始化编码器、解码器和瓶颈层的所有层。

2.前向传播方法 (forward):
输入:
x: 输入图像，形状为 (Batch_Size, 4, Height / 8, Width / 8)。
context: 上下文嵌入，形状为 (Batch_Size, Seq_Len, Dim)。
time: 时间步嵌入，形状为 (1, 1280)。

编码过程:
将输入 x 通过编码器的所有层，依次处理并保存每层的输出作为跳跃连接。

瓶颈处理:
将编码器输出的 x 通过瓶颈层的所有层进行处理。

解码过程:
遍历解码器的所有层：对于每个解码器层，将当前 x 与对应的跳跃连接输出进行拼接（拼接后的通道数增加）。
将拼接后的 x 通过当前解码器层进行处理。

输出: 返回解码器最终输出的特征图。

"""

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化编码器，由多个 SwitchSequential 模块组成
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        # 初始化瓶颈层，由多个层组成
        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )

        # 初始化解码器，由多个 SwitchSequential 模块组成
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x



"""
UNET_OutputLayer
UNET_OutputLayer 类是 UNet 架构中的输出层，负责将最终的解码器特征图转换为所需的输出格式。
它通过组归一化（Group Normalization）、激活函数（SiLU）和卷积层，将高维特征图映射到低维度的输出空间。

UNET_OutputLayer 类通过以下步骤实现特征图的输出转换：
1.组归一化: 对输入特征图进行组归一化处理，以稳定训练过程。
2.激活函数: 应用 SiLU 激活函数，增加非线性特性。
3.卷积层: 通过卷积层将特征图的通道数从 320 转换为 4，生成最终输出。
这种结构在图像分割任务中常用于将高维特征图转换为低维度的输出空间，例如生成分割掩码或预测图像的像素级别信息。

一 初始化参数:
in_channels (int): 输入特征的通道数。
out_channels (int): 输出特征的通道数。

二 主要组件:
self.groupnorm: 组归一化层，用于对输入特征图进行归一化，组数为32。
self.conv: 卷积层，将输入特征的通道数从 in_channels 转换为 out_channels，卷积核大小为3，填充为1。


详细步骤:
1.初始化:
初始化组归一化层和卷积层。

2.前向传播方法 (forward):
输入: x 的形状为 (Batch_Size, 320, Height / 8, Width / 8)。
组归一化: 对输入 x 应用组归一化，保持形状不变。
激活函数: 对归一化后的 x 应用 SiLU 激活函数，保持形状不变。
卷积层: 对激活后的 x 应用卷积层，将通道数从 320 转换为 4，输出形状为 (Batch_Size, 4, Height / 8, Width / 8)。
输出: 返回处理后的特征图。

"""

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 初始化组归一化层，组数为32
        self.groupnorm = nn.GroupNorm(32, in_channels)
        # 初始化卷积层，将输入通道数从 in_channels 转换为 out_channels，卷积核大小为3，填充为1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)
        # 输入 x 的形状: (Batch_Size, 320, Height / 8, Width / 8)

        # 对输入 x 应用组归一化，保持形状不变
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        # 对归一化后的 x 应用 SiLU 激活函数，保持形状不变
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        # 对激活后的 x 应用卷积层，将通道数从 320 转换为 4，输出形状为 (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        # 返回处理后的特征图
        return x



"""
Diffusion
Diffusion 类实现了扩散模型（Diffusion Model）的核心组件，用于从噪声逐步生成目标数据（如图像）。
该模型通过时间嵌入（time embedding）、UNet 架构和输出层，将初始的潜在空间数据逐步转换为最终的输出。

Diffusion 类通过以下步骤实现扩散模型的核心功能：
1.时间嵌入: 将输入的时间步信息通过 TimeEmbedding 层进行嵌入，转换为更高维度的空间。
2.UNet 处理: 将嵌入后的时间信息、潜在空间数据和上下文信息输入到 UNet 中，进行特征提取和空间依赖关系捕捉。
3.输出转换: 将 UNet 的高维输出通过 UNET_OutputLayer 转换为低维度的输出空间，生成最终的特征图。
这种结构在图像生成任务中非常有效，能够逐步从噪声中生成高质量的图像。

一 初始化参数:
二 主要组件:
时间嵌入层:
self.time_embedding: TimeEmbedding 类实例，将时间步信息（1维，320维）嵌入到更高维度的空间（1280维）。

UNet 架构:
self.unet: UNET 类实例，用于处理输入的潜在空间数据和上下文信息，捕捉特征的空间依赖关系。

输出层:
self.final: UNET_OutputLayer 类实例，将 UNet 的输出从高维特征图转换为低维度的输出空间。


详细步骤:
1.初始化:
初始化时间嵌入层 TimeEmbedding，输入维度为320，输出维度为1280。
初始化 UNet 架构 UNET，用于处理和生成特征图。
初始化输出层 UNET_OutputLayer，将 UNet 的输出从320通道转换为4通道。

2.前向传播方法 (forward):
输入:
latent: 初始潜在空间数据，形状为 (Batch_Size, 4, Height / 8, Width / 8)。
context: 上下文嵌入，形状为 (Batch_Size, Seq_Len, Dim)。
time: 时间步信息，形状为 (1, 320)。

时间嵌入:
将时间步信息 time 通过 TimeEmbedding 层进行嵌入，输出形状从 (1, 320) 转换为 (1, 1280)。

UNet 处理:
将嵌入后的时间信息 time、潜在空间数据 latent 和上下文信息 context 输入到 UNet 中进行处理，输出形状为 (Batch, 320, Height / 8, Width / 8)。

输出层:
将 UNet 的输出通过 UNET_OutputLayer 层进行转换，将通道数从320减少到4，输出形状为 (Batch, 4, Height / 8, Width / 8)。

输出: 返回最终生成的特征图。

"""

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化时间嵌入层，将时间步信息从320维嵌入到1280维
        self.time_embedding = TimeEmbedding(320)
        # 初始化 UNet 架构，用于处理潜在空间数据和上下文信息
        self.unet = UNET()
        # 初始化输出层，将 UNet 的输出从320通道转换为4通道
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)
        # 输入 latent 的形状: (Batch_Size, 4, Height / 8, Width / 8)
        # 输入 context 的形状: (Batch_Size, Seq_Len, Dim)
        # 输入 time 的形状: (1, 320)

        # 将时间步信息 time 通过时间嵌入层进行嵌入，输出形状从 (1, 320) 转换为 (1, 1280)
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        # 将嵌入后的时间信息 time、潜在空间数据 latent 和上下文信息 context 输入到 UNet 中进行处理
        # 输出形状为 (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        # 将 UNet 的输出通过输出层进行转换，将通道数从320减少到4，输出形状为 (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        # 返回最终生成的特征图
        return output
    
