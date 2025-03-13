import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate time series data (1-minute intervals)
n_samples = 2000  # Generate 2000 samples
start_time = datetime(2024, 1, 1, 12, 0)  # Start time
timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]

# Function to generate a Fourier-based signal
def generate_fourier_signal(n, frequencies, amplitudes):
    t = np.linspace(0, 1, n)
    signal = np.zeros(n)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))  # Normalize to [0,1]

# Generate synthetic data using Fourier signals
frequencies = [0.5, 1.2, 2.5]  # Hz
amplitudes = [0.7, 0.5, 0.3]

topspin_backspin = (generate_fourier_signal(n_samples, frequencies, amplitudes) > 0.5).astype(int)  # Binary
forehand_backhand = (generate_fourier_signal(n_samples, frequencies, amplitudes) > 0.5).astype(int)  # Binary
winning_in_first_3_strokes = (generate_fourier_signal(n_samples, frequencies, amplitudes) > 0.6).astype(int)  # Binary

ball_position_x = (generate_fourier_signal(n_samples, [0.3, 1.0, 2.0], [1.0, 0.5, 0.2]) * 152).astype(int)  # Integer range [1,152]
ball_position_y = (generate_fourier_signal(n_samples, [0.4, 1.5, 2.8], [0.8, 0.6, 0.3]) * 140).astype(int)  # Integer range [1,140]

# Create DataFrame
df = pd.DataFrame({
    "Time": timestamps,
    "Topspin/Backspin Indicator": topspin_backspin,
    "Forehand/Backhand Indicator": forehand_backhand,
    "Winning in First Three Strokes": winning_in_first_3_strokes,
    "X": ball_position_x, # Ball Position X
    "Y": ball_position_y # Ball Position Y
})

# Keep only the last 1500 records
df = df.tail(1500)

# Export data to CSV
df.to_csv("table_tennis_data.csv", index=False)

# Plot each column's trend
df.set_index("Time", inplace=True)
plt.figure(figsize=(12, 6))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Time-Series Trends of Table Tennis Data")
plt.legend()
plt.grid(True)
plt.show()
