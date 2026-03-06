import torch

class CharacterTokenizer:
    """
    字符级分词器：将每一个唯一的字符映射为一个唯一的整数索引
    """
    def __init__(self, text: str):
        # 1. 提取文本中所有不重复的字符并排序，确保索引的稳定性
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 2. 创建映射表：stoi (string to index) 和 itos (index to string)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s: str):
        """ 将字符串转化为整数列表 """
        # 如果字符不在词表中，这里会报错，这是为了保证训练数据的纯净性
        return [self.stoi[c] for c in s]

    def decode(self, l: list):
        """ 将整数列表还原为字符串 """
        return ''.join([self.itos[i] for i in l])

    def encode_as_tensor(self, s: str):
        """ 直接输出为 PyTorch 张量，方便模型直接读取 """
        return torch.tensor(self.encode(s), dtype=torch.long)

# 示例：
# tokenizer = CharacterTokenizer("你好，Moon！")
# print(tokenizer.vocab_size) # 输出词表大小