from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 基础维度
    block_size: int = 128     # 上下文窗口：模型一次能看多少个字符
    n_embd: int = 256         # 词嵌入维度：每个字符被转化为多长的向量
    
    # 深度与广度
    n_head: int = 4           # 多头注意力的头数
    n_layer: int = 4          # Transformer Block 堆叠的层数
    
    # 训练稳定性
    dropout: float = 0.2      # 丢弃率：随机“遗忘”一些神经元，防止死记硬背
    
    # 词表信息（由 Tokenizer 确定）
    vocab_size: int = 65      # 我们先给一个默认值，训练时会根据数据动态更新