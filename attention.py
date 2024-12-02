import torch
from torch import nn
from torch.nn import functional as F
import math


"""
SelfAttention 自注意力机制
SelfAttention 类实现了自注意力机制，用于计算序列中每个元素与其他元素之间的关系，从而捕捉序列内部的依赖关系。

一 初始化参数: n_heads, d_embed, in_proj_bias=True, out_proj_bias=True
二 主要组件: self.in_proj, self.out_proj, self.n_heads, self.d_head
三 前向传播方法 (forward)

详细步骤：
1.输入投影：
将输入 x 通过线性层 self.in_proj 投影到 Q, K, V 三个向量，每个向量的维度为 d_embed。使用 chunk(3, dim=-1) 将投影后的向量分割成 Q, K, V。

2.调整形状以适应多头注意力：
将 Q, K, V 的形状从 (Batch_Size, Seq_Len, Dim) 调整为 (Batch_Size, Seq_Len, H, Dim/H)，然后转置为 (Batch_Size, H, Seq_Len, Dim/H)，以便进行多头注意力计算。

3.计算注意力权重：
计算 Q 和 K 的点积，得到注意力权重 weight，形状为 (Batch_Size, H, Seq_Len, Seq_Len)。

4.应用因果掩码（可选）：
如果 causal_mask 为 True，则使用上三角掩码将上三角部分的注意力权重设为 -inf，以防止模型看到未来的信息。这在自回归模型（如 GPT）中常用。

5.缩放和 softmax：
对注意力权重进行缩放，除以 sqrt(d_head)，以防止数值不稳定。应用 softmax 激活函数，将权重归一化。

6.计算最终输出：
将注意力权重与 V 相乘，得到加权后的值 output，形状为 (Batch_Size, H, Seq_Len, Dim/H)。调整形状为 (Batch_Size, Seq_Len, Dim)。通过线性层 self.out_proj 投影回原始的嵌入维度。

"""

class SelfAttention(nn.Module):
    '''
    n_heads(int): 注意力头的数量。将输入的嵌入维度分成多个头，每个头独立计算注意力。
    d_embed(int): 输入嵌入的维度大小。
    in_proj_bias(bool): 是否在输入投影层中添加偏置项。默认为 True。
    out_proj_bias(bool): 是否在输出投影层中添加偏置项。默认为 True。
    '''

    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        # 将输入的嵌入向量线性投影到查询（Query）、键（Key）和值（Value）三个向量。
        # 线性层的输出维度为 3 * d_embed，因为它同时生成 Q、K、V 三个向量。
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)

        # This one represents the Wo matrix
        # 将注意力机制输出的结果线性投影回原始的嵌入维度 d_embed。
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        # 注意力头的数量
        self.n_heads = n_heads

        # 每个注意力头的维度大小，计算方式为 d_embed // n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: # (Batch_Size, Seq_Len, Dim)
        # 输入 x 的形状: (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        # 重新调整形状以适应多头注意力
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)  

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        # 输入投影：将 x 投影到 Q, K, V
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # 每个的形状: (Batch_Size, Seq_Len, Dim)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        # 调整 Q, K, V 的形状以适应多头注意力
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        # 计算注意力权重：Q @ K^T
        weight = q @ k.transpose(-1, -2)
        
        # 如果使用因果掩码，则将上三角部分设为负无穷
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)  # 上三角为 True
            # Fill the upper triangle with -inf
            # 上三角填充为 -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        # 缩放注意力权重
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        # 应用 softmax 激活函数
        weight = F.softmax(weight, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        # 计算注意力输出：weight @ V
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        # 调整输出形状
        output = output.transpose(1, 2) 

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # 线性投影输出
        output = self.out_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output
    


"""
CrossAttention 交叉注意力机制
CrossAttention 类实现了交叉注意力机制，用于在两个不同的序列之间建立关联。

一 初始化参数: n_heads, d_embed, d_cross, in_proj_bias, out_proj_bias
二 主要组件: self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.n_heads, self.d_head
三 前向传播方法 (forward)

详细步骤：
1.输入投影：
将查询序列 x 通过线性层 self.q_proj 投影到 Q，维度为 d_embed。
将上下文序列 y 通过线性层 self.k_proj 和 self.v_proj 分别投影到 K 和 V，维度均为 d_embed。

2.调整形状以适应多头注意力：
将 Q, K, V 的形状从 (Batch_Size, Seq_Len, Dim) 调整为 (Batch_Size, H, Seq_Len, Dim/H)，以便进行多头注意力计算。

3.计算注意力权重：
计算 Q 和 K 的点积，得到注意力权重 weight，形状为 (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)。

4.缩放和 softmax：
对注意力权重进行缩放，除以 sqrt(d_head)。应用 softmax 激活函数，将权重归一化。

5.计算最终输出：
将注意力权重与 V 相乘，得到加权后的值 output，形状为 (Batch_Size, H, Seq_Len_Q, Dim_Q/H)。
调整形状为 (Batch_Size, Seq_Len_Q, Dim_Q)。通过线性层 self.out_proj 投影回原始的嵌入维度。

"""

class CrossAttention(nn.Module):

    '''
    n_heads(int): 注意力头的数量。
    d_embed(int): 查询(Query)嵌入的维度大小。
    d_cross(int): 键(Key)和值(Value)嵌入的维度大小。
    in_proj_bias(bool): 是否在 Q, K, V 投影层中添加偏置项。默认为 True。
    out_proj_bias(bool): 是否在输出投影层中添加偏置项。默认为 True。
    '''

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # 将查询序列 x 投影到查询向量 Q，维度为 d_embed
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # 将上下文序列 y 投影到键向量 K，维度为 d_embed
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # 将上下文序列 y 投影到值向量 V，维度为 d_embed
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # 将注意力机制的输出线性投影回原始的嵌入维度 d_embed
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        # 注意力头的数量
        self.n_heads = n_heads
        # 每个注意力头的维度大小，计算方式为 d_embed // n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        # 查询 x 的形状: (Batch_Size, Seq_Len_Q, Dim_Q)
        # 上下文 y 的形状: (Batch_Size, Seq_Len_KV, Dim_KV)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        # 调整形状以适应多头注意力
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # 投影 Q, K, V
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(x)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)

        # 调整 Q, K, V 的形状以适应多头注意力
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2) 
        
        # 计算注意力权重：Q @ K^T
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        # 缩放注意力权重
        weight /= math.sqrt(self.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        # 应用 softmax 激活函数
        weight = F.softmax(weight, dim=-1)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        # 计算注意力输出：weight @ V
        output = weight @ v
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        # 调整输出形状
        output = output.transpose(1, 2).contiguous()
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        # 线性投影输出
        output = self.out_proj(output)

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
