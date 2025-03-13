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
frequencies = [0.5, 1.2, 2.5]
amplitudes = [0.7, 0.5, 0.3]

topspin_backspin = (generate_fourier_signal(n_samples, frequencies, amplitudes) > 0.5).astype(int)
forehand_backhand = (generate_fourier_signal(n_samples, frequencies, amplitudes) > 0.5).astype(int)
winning_in_first_3_strokes = (generate_fourier_signal(n_samples, frequencies, amplitudes) > 0.6).astype(int)

# Generate X, Y positions
ball_position_x = (generate_fourier_signal(n_samples, [0.3, 1.0, 2.0], [1.0, 0.5, 0.2]) * 152).astype(int) + 1
ball_position_y = (generate_fourier_signal(n_samples, [0.4, 1.5, 2.8], [0.8, 0.6, 0.3]) * 140).astype(int) + 1

# Add random noise to X, Y to scatter the values
noise_x = np.random.randint(-10, 10, size=n_samples)
noise_y = np.random.randint(-10, 10, size=n_samples)

ball_position_x = np.clip(ball_position_x + noise_x, 1, 152)
ball_position_y = np.clip(ball_position_y + noise_y, 1, 140)


# Convert X, Y into 3×3 Landing Zone (4-9)
def calculate_landing_zone(x, y):
    x_bin = np.clip(np.ceil(x / (152 / 3)).astype(int), 1, 3)
    y_bin = np.clip(np.ceil(y / (140 / 3)).astype(int), 2, 3)

    # Original mapping (1-9)
    original_zone = (y_bin - 1) * 3 + x_bin

    # Map 1-3 → 4-6, 4-6 → 7-9
    new_zone_mapping = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9}

    return new_zone_mapping.get(original_zone, None)


landing_zone = np.array([calculate_landing_zone(x, y) for x, y in zip(ball_position_x, ball_position_y)])

# Filter out None values (invalid landing zones)
valid_indices = landing_zone != None
df = pd.DataFrame({
    "Time": np.array(timestamps)[valid_indices],
    "Topspin/Backspin Indicator": topspin_backspin[valid_indices],
    "Forehand/Backhand Indicator": forehand_backhand[valid_indices],
    "Winning in First Three Strokes": winning_in_first_3_strokes[valid_indices],
    "Landing Zone": landing_zone[valid_indices].astype(int)
})

# Add random perturbation to Landing Zone (simulate realistic variation)
random_shift = np.random.choice([-1, 0, 1], size=len(df), p=[0.2, 0.6, 0.2])
df["Landing Zone"] = np.clip(df["Landing Zone"] + random_shift, 4, 9)

# Ensure values remain within the range 4-9
assert df["Landing Zone"].between(4, 9).all(), "Landing Zone out of range!"

# Keep only the last 1500 records
#df = df.tail(1500)

# Export data to CSV
df.to_csv("table_tennis_data.csv", index=False)

# Plot trend of Landing Zone
df.set_index("Time", inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Landing Zone"], label="Landing Zone", marker='o', linestyle='-', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Landing Zone (4-9)")
plt.title("Time-Series Trends of Landing Zones (More Scattered)")
plt.legend()
plt.grid(True)
plt.show()
