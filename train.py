import torch
from model import MoonLanguageModel, ModelConfig
from scripts.tokenizer import CharacterTokenizer

# --- 1. 环境准备与超参数 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 优先使用 GPU 
batch_size = 64        # 每次训练的句子数量
max_iters = 10000       # 总训练步数
eval_interval = 500    # 每隔多久评估一次模型
learning_rate = 3e-4   # 寻觅真理的步长
eval_iters = 200       # 评估时的采样次数

# --- 2. 加载数据与分词器 ---
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)
data = tokenizer.encode_as_tensor(text)

# 划分训练集与验证集 (90% 训练, 10% 验证)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- 3. 初始化配置 ---
config = ModelConfig(vocab_size=tokenizer.vocab_size)

# 数据批处理函数
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    # 随机选择起始位置
    ix = torch.randint(0, len(data_source) - config.block_size, (batch_size,))
    x = torch.stack([data_source[i:i+config.block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+config.block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 4. 初始化模型与优化器 ---
model = MoonLanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    """ 评估函数：平滑损失值，观察模型是否过拟合 """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 5. 核心训练循环 ---
print(f"开始训练... 运行设备: {device}")
for iter in range(max_iters):

    # 定期评估与日志打印
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"步数 {iter}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")

    # 获取数据并计算梯度
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True) # 清空旧梯度
    loss.backward()                      # 反向传播（计算偏导数）
    optimizer.step()                     # 更新参数

# --- 6. 保存我们努力的成果 ---
torch.save(model.state_dict(), 'model/moon_model.pth')
print("训练完成，模型已封印在 model/moon_model.pth (红瞳微眯)")