import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


"""
CLIPEmbedding 嵌入-将 token 转换为嵌入向量，并添加位置嵌入。
CLIPEmbedding 类负责将输入的 token（通常是文本中的单词或子词）转换为嵌入向量（embeddings），
并添加位置嵌入（positional embeddings）以捕捉序列中 token 的位置信息。

一 初始化参数: n_vocab, n_embd, n_token
二 主要组件: self.token_embedding, self.position_embedding
三 前向传播方法 (forward)

详细步骤：
1.Token 嵌入：
输入 tokens 是一个形状为 (Batch_Size, Seq_Len) 的张量，包含每个 token 的索引。
通过 self.token_embedding 将每个 token 索引转换为对应的嵌入向量，输出形状为 (Batch_Size, Seq_Len, Dim)。

2.位置嵌入：
self.position_embedding 是一个形状为 (n_token, n_embd) 的可学习参数矩阵，包含每个位置的位置嵌入。
通过 self.position_embedding[:tokens.size(1), :] 仅选择当前序列长度对应的位置嵌入（因为位置嵌入的长度可能小于 n_token）。
将位置嵌入加到 token 嵌入上，得到最终的嵌入向量 x，形状仍为 (Batch_Size, Seq_Len, Dim)。

"""

class CLIPEmbedding(nn.Module):
    '''
    n_vocab（int）：词汇表的大小，即不同 token 的数量。
    n_embd（int）：嵌入向量的维度大小。
    n_token（int）：序列中 token 的最大长度，用于初始化位置嵌入。
    '''

    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        # 将输入的 token 索引转换为嵌入向量。参数 n_vocab 和 n_embd 分别指定了词汇表的大小和嵌入向量的维度。
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        # 一个可学习的参数矩阵，用于编码每个 token 的位置信息。形状为 (n_token, n_embd)，即每个位置对应一个嵌入向量。
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        # 输入 tokens 的形状: (Batch_Size, Seq_Len)
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding  # 仅选择当前序列长度对应的位置嵌入
        
        return x
    


"""
CLIPLayer 实现一个 Transformer 编码器层
CLIPLayer 类实现了一个 Transformer 编码器层，包括多头自注意力机制和前馈神经网络（Feed-Forward Network, FFN）。
每个编码器层都包含残差连接（residual connection）和层归一化（Layer Normalization）。

一 初始化参数: n_head, n_embd
二 主要组件: self.layernorm_1, self.attention, self.layernorm_2, self.linear_1, self.linear_2, QuickGELU
三 前向传播方法 (forward)

详细步骤：
1.自注意力机制：
残差连接：存储输入 x 到 residue，以便后续进行残差连接。
层归一化：对输入 x 进行层归一化，输出形状仍为 (Batch_Size, Seq_Len, Dim)。
自注意力：调用 SelfAttention 模块计算自注意力，输出形状为 (Batch_Size, Seq_Len, Dim)。
残差连接：将注意力输出与 residue 相加，实现残差连接。

2.前馈神经网络：
残差连接：再次存储输入 x 到 residue。
层归一化：对输入 x 进行层归一化。
线性层：通过 self.linear_1 将维度扩展到 4 * n_embd。
QuickGELU 激活函数：使用 torch.sigmoid(1.702 * x) 作为激活函数，模拟 GELU 的效果。
线性层：通过 self.linear_2 将维度恢复到 n_embd。
残差连接：将前馈神经网络的输出与 residue 相加。

"""

class CLIPLayer(nn.Module):
    '''
    n_head（int）：注意力头的数量。
    n_embd（int）：嵌入向量的维度大小。   
    '''

    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Pre-attention norm
        # 对输入进行层归一化，用于自注意力层之前。
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        # 多头自注意力机制，接收嵌入向量 x 并输出加权后的向量。
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        # 对输入进行层归一化，用于前馈神经网络之前。
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        # 前馈神经网络的第一层，线性层将维度从 n_embd 扩展到 4 * n_embd
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # 前馈神经网络的第二层，线性层将维度从 4 * n_embd 恢复到 n_embd
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        # 输入 x 的形状: (Batch_Size, Seq_Len, Dim)
        residue = x  # 存储残差连接的输入
        
        ### SELF ATTENTION 自注意力机制 ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # 层归一化
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # 自注意力机制
        x = self.attention(x, causal_mask=True)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # 残差连接
        x += residue

        ### FEEDFORWARD LAYER 前馈神经网络###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x  # 存储残差连接的输入
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # 层归一化
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        # 前馈神经网络第一层
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        # QuickGELU 激活函数
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        # 前馈神经网络第二层
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        # 残差连接
        x += residue

        return x



"""
CLIP
CLIP 类是整个模型的主干，类似于 CLIP 模型的编码器部分。
它将输入的 token 序列通过嵌入层、多个 Transformer 编码器层和层归一化层进行处理，最终输出上下文感知的嵌入向量。

一 初始化参数: 无显式参数，因为所有组件都在 __init__ 方法中定义。
二 主要组件: self.embedding, self.layers, self.layernorm
三 前向传播方法 (forward)

详细步骤：
1.嵌入和位置嵌入：
输入 tokens 是一个形状为 (Batch_Size, Seq_Len) 的张量，包含每个 token 的索引。
通过 self.embedding 将 token 转换为嵌入向量，并添加位置嵌入，输出形状为 (Batch_Size, Seq_Len, Dim)。

2.Transformer 编码器层：
将嵌入向量 state 通过多个 CLIPLayer 进行处理。每个编码器层都包含自注意力机制和前馈神经网络。
每一层的输出作为下一层的输入，最终得到上下文感知的嵌入向量。

3.最终层归一化：
对编码器层的输出进行层归一化，得到最终的输出 output，形状为 (Batch_Size, Seq_Len, Dim)。

"""

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # 将输入的 token 转换为嵌入向量，并添加位置嵌入。
        self.embedding = CLIPEmbedding(49408, 768, 77)

        # 包含多个 CLIPLayer，每个层都是一个 Transformer 编码器层。默认情况下，定义了 12 个编码器层。
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        # 对编码器层的输出进行最终的层归一化。
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # 应用多个 Transformer 编码器层
        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        
        # 最终的层归一化
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output
