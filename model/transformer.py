import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import MultiHeadAttention, FeedForward

class Block(nn.Module):
    """ 整合了层归一化、自注意力和前馈网络的标准 Block """
    def __init__(self, config):
        super().__init__()
        # 计算每个头的维度：n_embd 必须能被 n_head 整除哦
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_embd, config.n_head, head_size, config.block_size, config.dropout)
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # 使用 Pre-norm 架构，这在深层网络中更稳定
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MoonLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 核心权重表
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # 堆叠 Transformer Blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # 别忘了给 config 留个备份，推理时会用到
        self.config = config

        # 优雅的权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 将输入转化为高维空间的律动
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # 融合内容与位置
        
        x = self.blocks(x)    
        x = self.ln_f(x)      
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss