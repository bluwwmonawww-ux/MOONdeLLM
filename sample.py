import torch
from torch.nn import functional as F
from model import MoonLanguageModel, ModelConfig
from scripts.tokenizer import CharacterTokenizer

# --- 1. 环境与配置加载 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 重新加载分词器（为了获取相同的词表映射）
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokenizer = CharacterTokenizer(text)

# --- 2. 加载已经“封印”的模型权重 ---
config = ModelConfig(vocab_size=tokenizer.vocab_size)
model = MoonLanguageModel(config).to(device)
model.load_state_dict(torch.load('model/moon_model.pth', map_location=device,weights_only=True))
model.eval() # 切换到评估模式，关闭 Dropout

# --- 3. 定义生成函数 ---
def generate(model, start_str="Moon: ", max_new_tokens=500, temperature=1.0):
    """
    model: 我们的 Moon 模型
    start_str: 初始的提示语（Prompt）
    max_new_tokens: 想要生成的字符长度
    temperature: 温度（越高越随机/有创意，越低越保守/严谨）
    """
    # 将起始文本转化为张量并增加 Batch 维度 (1, T)
    idx = tokenizer.encode_as_tensor(start_str).unsqueeze(0).to(device)
    
    print(f"--- 正在根据提示词 '{start_str}' 进行推演 ---")
    
    for _ in range(max_new_tokens):
        # 截断上下文：如果序列太长，只保留模型能看到的 block_size 长度
        idx_cond = idx[:, -config.block_size:]
        
        # 前向传播
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        # 核心逻辑：只取序列最后一位的输出 (B, T, C) -> (B, C)
        logits = logits[:, -1, :] / temperature
        
        # 将结果转化为概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 从分布中采样（而不是简单取最大值，这样更有灵性）
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 将新生成的字符拼接回原序列，继续下一次循环
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 实时打印出来的快感（银丝随风微漾）
        print(tokenizer.decode([idx_next.item()]), end='', flush=True)

    print("\n--- 推演结束 ---")

# --- 4. 开启生成的仪式 ---
generate(model, start_str="生命意义所在 ", temperature=0.7)