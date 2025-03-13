import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.seasonal import STL

def detect_top_seasonal_periods(data, sampling_rate=1, top_k=2):
    """
    Use FFT to detect the top seasonal periods in a time series.

    Parameters:
        data: np.array, input time series
        sampling_rate: float, sampling interval (default = 1, assuming evenly spaced data)
        top_k: int, number of top seasonal periods to extract (default = 2)

    Returns:
        top_periods: list of int, detected top K seasonal periods
    """
    N = len(data)  # Number of data points
    freqs = fftfreq(N, d=sampling_rate)  # Compute frequency values
    fft_values = np.abs(fft(data))  # Compute FFT magnitudes (absolute values)

    # Keep only positive frequencies (ignore negative and DC component)
    pos_freqs = freqs[freqs > 0]
    pos_fft_values = fft_values[freqs > 0]

    # Find the indices of the top K dominant frequencies
    top_indices = np.argsort(pos_fft_values)[-top_k:][::-1]  # Sort in descending order

    # Compute corresponding periods T = 1 / f
    top_periods = [int(round(1 / pos_freqs[i])) for i in top_indices if pos_freqs[i] > 0]

    return top_periods  # Return the detected seasonal periods

# ✅ Generate sample time series data (two main periods: 50 and 20)
np.random.seed(42)
L = 300  # Time series length
time = np.arange(L)

# Define multiple frequency components
X_t = (
    5 * np.sin(2 * np.pi * time / 50) +  # Period 50
    3 * np.sin(2 * np.pi * time / 20) +  # Period 20
    4 * np.sin(2 * np.pi * time / 35) +  # Additional Period 35
    2 * np.sin(2 * np.pi * time / 15) +  # Additional Period 15
    np.random.normal(0, 1, L)            # Random noise
)

# ✅ Detect the top 2 seasonal periods using FFT
top_periods = detect_top_seasonal_periods(X_t, top_k=2)
print(f"🔍 Detected Top 2 Seasonal Periods: {top_periods}")

# ✅ Generate reconstructed signals using top 2 periods
reconstructed_signals = [np.sin(2 * np.pi * time / period) for period in top_periods]

# ✅ Plot the time series data along with top 2 Fourier components
plt.figure(figsize=(10, 6))

# Plot original time series
plt.plot(time, X_t, label="Original Time Series", color="black", alpha=0.6)

# Plot top 2 detected Fourier components
for i, (period, signal) in enumerate(zip(top_periods, reconstructed_signals)):
    plt.plot(time, signal * np.std(X_t), label=f"Fourier Component {i+1} (Period = {period})", linestyle="dashed")

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Time Series with Multiple Frequencies and Noise")
plt.legend()
plt.show()

#==========================

# ✅ Perform STL decomposition
stl = STL(X_t, period=top_periods[0])  # Use the dominant period for decomposition
res = stl.fit()

# Extract trend, seasonal, and residual components
trend = res.trend
seasonal = res.seasonal
residual = res.resid

# ✅ Plot original data along with STL components
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Plot original time series
axes[0].plot(time, X_t, label="Original Time Series", color="black")
axes[0].set_title("Original Time Series")
axes[0].legend()

# Plot trend component
axes[1].plot(time, trend, label="Trend Component", color="blue")
axes[1].set_title("Trend Component (XT)")
axes[1].legend()

# Plot seasonal component
axes[2].plot(time, seasonal, label="Seasonal Component", color="green")
axes[2].set_title("Seasonal Component (XS)")
axes[2].legend()

# Plot residual component
axes[3].plot(time, residual, label="Residual Component", color="red")
axes[3].set_title("Residual Component (XR)")
axes[3].legend()

plt.xlabel("Time")
plt.tight_layout()
plt.show()
