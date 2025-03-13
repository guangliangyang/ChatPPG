import numpy as np
import matplotlib.pyplot as plt

# 模型名称
llm_models = ["GPT-2", "Llama-2-7b", "Llama-3.2-1B", "DeepSeek-R1-Distill-Qwen-1.5B"]
ablation_models = ["ChatPPG", "w/o LLM", "w/o Freq Pro", "w/o HumKnow", "w/o IChannel", "w/o FltProj", "w/o IEF"]

# MAE/MSE 数据
mae_values = {
    "ChatPPG": [0.522, 0.514, 0.493, 0.441],
    "w/o LoRA": [0.541, 0.533, 0.510, 0.457],
    "w/o Freq Pro": [0.541, 0.529, 0.501, 0.468],
    "w/o HumKnow": [0.598, 0.593, 0.569, 0.527],
    "w/o IChannel": [0.594, 0.585, 0.560, 0.512],
    "w/o FltProj": [0.557, 0.547, 0.522, 0.470],
    "w/o IEF": [0.618, 0.613, 0.591, 0.539]
}


mse_values = {
    "ChatPPG": [0.512, 0.503, 0.475, 0.432],
    "w/o LoRA": [0.520, 0.525, 0.491, 0.442],
    "w/o Freq Pro": [0.529, 0.518, 0.482, 0.456],
    "w/o HumKnow": [0.591, 0.581, 0.553, 0.511],
    "w/o IChannel": [0.585, 0.577, 0.541, 0.503],
    "w/o FltProj": [0.548, 0.531, 0.502, 0.462],
    "w/o IEF": [0.610, 0.607, 0.572, 0.527]
}

# 定义不同的线型、标记和颜色
line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
markers = ['o', 's', '^', 'd', 'v', '<', '>']
colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']

# 画 MAE 曲线
plt.figure(figsize=(10, 6))
for i, (ab_model, values) in enumerate(mae_values.items()):
    plt.plot(llm_models, values, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=ab_model)

plt.xlabel("LLM Models")
plt.ylabel("MAE")
plt.title("MAE Comparison of Ablation Experiments on Different LLMs")
plt.legend()
plt.grid(True)
plt.show()

# 画 MSE 曲线
plt.figure(figsize=(10, 6))
for i, (ab_model, values) in enumerate(mse_values.items()):
    plt.plot(llm_models, values, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=ab_model)

plt.xlabel("LLM Models")
plt.ylabel("MSE")
plt.title("MSE Comparison of Ablation Experiments on Different LLMs")
plt.legend()
plt.grid(True)
plt.show()
