import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    """ 多头自注意力机制：让模型同时关注不同位置的信息 """

    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        """block_size 是上下文窗口大小，dropout 是注意力权重的丢弃率"""
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        
        # 核心投影矩阵：将输入映射到 Q, K, V
        # 我们这里一次性算出所有头的 QKV，效率更高
        self.key = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.query = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.value = nn.Linear(n_embd, n_head * head_size, bias=False)
        
        # 因果遮盖（Mask）：防止模型“偷看未来”
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # 输出投影与 Dropout（防止过拟合的遗忘机制）
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # Batch size, Time (sequence length), Channels (n_embd)
        
        # 1. 计算 Q, K, V 并进行分头操作
        # 转置后的形状：(B, n_head, T, head_size)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 2. 计算注意力权重（Affinity）
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        
        # 应用 Mask：将未来的信息设为负无穷，Softmax 后变为 0
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 3. 加权求和并重新组合多个头
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)

        # 4. 最后经过一个输出线性层
        return self.resid_dropout(self.proj(y))
class FeedForward(nn.Module):
    """ 一个简单的线性层，紧随注意力机制之后 """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # 这里的 4 是经验法则：通常将维度放大 4 倍再缩小
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # 激活函数，引入非线性
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
