# 🌙 Moon Language Model

一个从零开始构建的 GPT-like 语言模型，实现了 Transformer 架构的核心组件。

## ✨ 特性

- **多头自注意力机制（Multi-Head Attention）** - 让模型同时关注不同位置的信息
- **前馈网络（Feed Forward Network）** - 非线性信息处理
- **因果遮盖（Causal Masking）** - 防止模型"偷看未来"的token
- **残差连接（Residual Connections）** - 让梯度顺畅流动
- **层归一化（Layer Normalization）** - 保持训练稳定性
- **字符级分词器（Character Tokenizer）** - 支持任意文本

## 🏗️ 项目结构

```
Moon_LLM/
├── model/
│   ├── __init__.py           # 模块导入
│   ├── attention.py          # 多头注意力和前馈网络
│   ├── transformer.py        # Block 和 MoonLanguageModel
│   └── config.py             # 模型配置
├── scripts/
│   └── tokenizer.py          # 字符级分词器
├── data/
│   └── input.txt             # 训练文本（需自行添加）
├── train.py                  # 训练脚本
├── sample.py                 # 文本生成脚本
└── requirements.txt          # 依赖包
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

在 `data/input.txt` 中放入你的训练文本。

### 3. 训练模型

```bash
python train.py
```

训练完成后，模型权重会保存到 `model/moon_model.pth`

### 4. 生成文本

```bash
python sample.py
```

修改 `sample.py` 最后一行的 `start_str` 和 `temperature` 参数来控制生成效果。

## 📊 模型配置

在 `model/config.py` 中调整：

```python
block_size = 128       # 上下文窗口大小
n_embd = 256          # 嵌入维度
n_head = 4            # 注意力头数
n_layer = 4           # Transformer 层数
dropout = 0.2         # Dropout 概率
```

## 💡 核心概念

### 多头自注意力（Multi-Head Attention）

让模型通过多个"角度"同时观察输入。每个头学习不同的关注模式。

### 因果遮盖（Causal Masking）

在生成文本时，当前位置的token只能看到过去和现在的token，不能看到未来的token。

### 位置编码（Position Encoding）

告诉模型每个token在序列中的位置，因为 Transformer 本身对位置顺序不敏感。

## 📈 性能指标

- 训练损失逐步下降
- 验证损失用于监控过拟合

## 🔧 故障排除

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError` | 检查 `model/__init__.py` 的导入 |
| CUDA 显存溢出 | 减小 `batch_size` 或 `n_layer` |
| 生成文本质量差 | 增加训练步数或模型容量 |

## 📚 学习资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 论文

## 💬 说明

这是一个教学项目，用于理解 Transformer 架构的工作原理。代码包含详细的中文注释，适合学习。

---

**作者**: Moon (with ❤️)  
**修改日期**: 2026年3月6日
