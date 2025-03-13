import matplotlib.pyplot as plt
import numpy as np

# Updated data from the image
llm_models = ["GPT-2", "Llama-2-7b", "Llama-3.2-1B", "DeepSeek-R1-Distill-Qwen-1.5B"]
time_series_models = ["ChatPPG (Ours)", "Time-LLM", "TEMPO", "Autotimes", "LLM-Time"]
data = {
    "ChatPPG (Ours)": [31.2, 37.4, 53.8, 121.6],
    "Time-LLM": [29.6, 35.8, 52.3, 120.3],
    "TEMPO": [31.2, 37.9, 54.1, 122.5],
    "Autotimes": [31.1, 37.2, 53.9, 121.7],
    "LLM-Time": [1362, 4757, 3673, 3976]
}
# Replot with improved color for LLM-Time
fig, ax1 = plt.subplots(figsize=(10, 6))

# Define colors and markers
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'd']
colors = ['b', 'g', 'r', 'm']

# Plot normal models on left y-axis
for i, (model, times) in enumerate(data.items()):
    if model != "LLM-Time":
        ax1.plot(llm_models, times, linestyle=line_styles[i % len(line_styles)],
                 marker=markers[i % len(markers)], color=colors[i % len(colors)], label=model)

ax1.set_xlabel("LLM Models")
ax1.set_ylabel("Inference Time (ms)", color='black')
ax1.set_title("Inference Time Comparison of Time-Series Models Across Different LLMs")
ax1.tick_params(axis='y', labelcolor='black')

# Create a second y-axis for LLM-Time (right side) with a new color
ax2 = ax1.twinx()
llm_time_color = "darkorange"  # Change to a more distinct color
ax2.plot(llm_models, data["LLM-Time"], linestyle="--", marker="x", color=llm_time_color, label="LLM-Time")
ax2.set_ylabel("LLM-Time Inference Time (ms)", color=llm_time_color)
ax2.tick_params(axis='y', labelcolor=llm_time_color)

# Add legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
