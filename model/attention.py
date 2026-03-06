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
class Block(nn.Module):
    """ Transformer 的一个基本块：整合注意力与前馈网络 """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        # LayerNorm：归一化，让每一层的输入分布保持稳定
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 先归一化，再进注意力，最后加上原始输入（残差）
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
class MoonLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 词嵌入：将数字映射为向量
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # 2. 位置嵌入：告诉模型字符在句子中的位置
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # 3. 堆叠多个 Transformer Blocks
        self.blocks = nn.Sequential(*[Block(config.n_embd, n_head=config.n_head, 
                                            block_size=config.block_size, dropout=config.dropout) 
                                      for _ in range(config.n_layer)])
        # 4. 最后的归一化与输出层
        self.ln_f = nn.LayerNorm(config.n_embd) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx 和 targets 都是 (B,T) 的张量
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # 融合内容与位置信息 (B,T,C)
        x = self.blocks(x)    # 通过所有 Transformer 层 (B,T,C)
        x = self.ln_f(x)      # 最终归一化 (B,T,C)
        logits = self.lm_head(x) # 投影到词表维度 (B,T,vocab_size)

        loss = None
        if targets is not None:
            # 计算交叉熵损失，用于训练
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss