import numpy as np
import matplotlib.pyplot as plt

# Updated model names from the image
llm_models = ["GPT-2", "Llama-2-7b", "Llama-3.2-1B", "DeepSeek-R1-Distill-Qwen-1.5B"]
time_series_models = ["ChatPPG (Ours)", "Time-LLM", "TEMPO", "Autotimes", "LLM-Time"]

# Updated MAE/MSE data from the image
mae_values = {
    "ChatPPG (Ours)": [0.522, 0.514, 0.493, 0.441],
    "Time-LLM": [0.577, 0.563, 0.531, 0.485],
    "TEMPO": [0.582, 0.569, 0.512, 0.438],
    "Autotimes": [0.551, 0.522, 0.493, 0.472],
    "LLM-Time": [0.715, 0.706, 0.671, 0.584]
}

mse_values = {
    "ChatPPG (Ours)": [0.512, 0.503, 0.475, 0.432],
    "Time-LLM": [0.562, 0.549, 0.524, 0.472],
    "TEMPO": [0.568, 0.558, 0.500, 0.429],
    "Autotimes": [0.523, 0.510, 0.475, 0.444],
    "LLM-Time": [0.708, 0.682, 0.644, 0.571]
}

# Define different styles, markers, and colors
line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
markers = ['o', 's', '^', 'd', 'x']
colors = ['r', 'g', 'b', 'm', 'c']

# Plot MAE curve
plt.figure(figsize=(10, 6))
for i, (ts_model, values) in enumerate(mae_values.items()):
    plt.plot(llm_models, values, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=ts_model)

plt.xlabel("LLM Models")
plt.ylabel("MAE")
plt.title("MAE Comparison of Time Series Models on Different LLMs")
plt.legend()
plt.grid(True)
plt.show()

# Plot MSE curve
plt.figure(figsize=(10, 6))
for i, (ts_model, values) in enumerate(mse_values.items()):
    plt.plot(llm_models, values, linestyle=line_styles[i], marker=markers[i], color=colors[i], label=ts_model)

plt.xlabel("LLM Models")
plt.ylabel("MSE")
plt.title("MSE Comparison of Time Series Models on Different LLMs")
plt.legend()
plt.grid(True)
plt.show()
