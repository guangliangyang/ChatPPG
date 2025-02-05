import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 数据集定义
class RockPaperScissorsDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Transformer 模型定义
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(100, embed_dim)  # 假定最大序列长度为100
        self.transformer = nn.Transformer(
            embed_dim, num_heads, num_layers, num_layers, hidden_dim
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_length = x.size(1)
        position = torch.arange(seq_length, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.position_embedding(position)
        x = x.permute(1, 0, 2)  # Transformer需要的输入是 (seq_len, batch_size, embed_dim)
        x = self.transformer(x, x)
        x = x.mean(dim=0)  # 聚合时间维度的信息
        return self.fc(x)

# 生成训练数据
def generate_data(num_samples=1000):
    actions = [0, 1, 2]  # 0: 石头, 1: 剪刀, 2: 布
    data = np.random.choice(actions, num_samples)
    return data

# 数据预处理
sequence_length = 5
data = generate_data(10000)
dataset = RockPaperScissorsDataset(data, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型初始化
vocab_size = 3  # 石头、剪刀、布
embed_dim = 16
num_heads = 2
hidden_dim = 64
num_layers = 2

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模型训练
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)  # 将数据移动到 GPU
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# 模型预测
def predict(model, history):
    model.eval()
    with torch.no_grad():
        history_tensor = torch.tensor(history, dtype=torch.long).unsqueeze(0).to(device)  # 批量维度
        output = model(history_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction

# 测试
history = [0, 1, 2, 0, 1]  # 假设最近五步为 石头->剪刀->布->石头->剪刀
#history ="458744544776975774537557487768568646478556"
prediction = predict(model, history)
action_map = {0: "石头", 1: "剪刀", 2: "布"}
print(f"模型预测下一步动作: {action_map[prediction]}")
