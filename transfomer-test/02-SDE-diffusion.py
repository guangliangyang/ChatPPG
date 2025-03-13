import torch
import torch.nn as nn
import numpy as np


class NoisePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NoisePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 输出维度与输入一致
        )

    def forward(self, x, t):
        t = t.view(-1, 1)  # 将时间步扩展为 (batch_size, 1)
        x_t = torch.cat([x, t], dim=-1)  # 拼接时间步
        return self.net(x_t)


def forward_diffusion(x, t, alpha_t):
    """
    扩散公式：q(x_t | x_{t-1}) = N(x_t; sqrt(alpha_t)*x, (1-alpha_t)*I)
    """
    noise = torch.randn_like(x)
    mean = torch.sqrt(alpha_t[t])[:, None] * x
    std = torch.sqrt(1 - alpha_t[t])[:, None]
    return mean + std * noise, noise


def train(model, optimizer, data, alpha_t, num_steps=1000):
    """
    训练神经网络 epsilon_theta
    """
    model.train()
    criterion = nn.MSELoss()
    for step in range(num_steps):
        # 随机采样时间步 t
        t = torch.randint(0, len(alpha_t), (data.size(0),)).long()

        # 正向扩散，得到 x_t 和真实噪声 epsilon
        x_t, noise = forward_diffusion(data, t, alpha_t)

        # 预测噪声
        t_float = t.float().to(data.device)
        predicted_noise = model(x_t, t_float)

        # 计算损失
        loss = criterion(predicted_noise, noise)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")


def generate_sequence(model, x_t, alpha_t, num_steps):
    """
    逆向生成序列
    """
    model.eval()
    with torch.no_grad():
        for t in range(num_steps - 1, -1, -1):
            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.float).to(x_t.device)
            pred_noise = model(x_t, t_tensor)
            mean = (1 / torch.sqrt(alpha_t[t])) * (x_t - (1 - alpha_t[t]) / torch.sqrt(1 - alpha_t[t]) * pred_noise)
            std = torch.sqrt(1 - alpha_t[t])

            # 采样
            noise = torch.randn_like(x_t) if t > 0 else 0  # 最后一步无噪声
            x_t = mean + std * noise
    return x_t


# 配置参数
input_dim = 100  # 输入序列维度
hidden_dim = 128
alpha_t = torch.linspace(0.01, 0.99, steps=100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 模型和优化器
model = NoisePredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(alpha_t.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 模拟训练数据
data = torch.sin(torch.linspace(0, 2 * np.pi, steps=100).unsqueeze(1)).to(alpha_t.device)  # 示例：正弦波序列

# 训练模型
train(model, optimizer, data, alpha_t)

# 使用训练好的模型生成序列
x_t = torch.randn_like(data)  # 从噪声开始
predicted_sequence = generate_sequence(model, x_t, alpha_t, num_steps=100)

print("Generated Sequence:", predicted_sequence)
