import numpy as np
import matplotlib.pyplot as plt

def fft_lowpass_filter(signal, cutoff):
    """
    Applies a low-pass filter in the frequency domain using FFT.
    
    Parameters:
      signal: 1D numpy array representing the noisy signal.
      cutoff: cutoff frequency index (integer) for the low-pass filter.
    
    Returns:
      The filtered signal as a 1D numpy array.
    """
    # Compute the FFT of the signal.
    X = np.fft.fft(signal)
    N = len(signal)
    
    # Create a low-pass filter mask: 1 for frequencies below cutoff, 0 for frequencies above.
    H = np.zeros(N)
    H[:cutoff] = 1
    H[-cutoff+1:] = 1  # Symmetry for negative frequencies.
    
    # Apply the filter in the frequency domain.
    Y = H * X
    
    # Compute the inverse FFT to get the filtered signal.
    y = np.fft.ifft(Y)
    return np.real(y)

# Example usage:
N = 256
t = np.linspace(0, 1, N)
# Create a clean signal: a low-frequency sine wave.
signal = np.sin(2 * np.pi * 5 * t)
# Add high-frequency noise.
noisy_signal = signal + 0.5 * np.random.randn(N)

# Apply the low-pass filter with a cutoff index of 20.
filtered_signal = fft_lowpass_filter(noisy_signal, cutoff=10)

# Plot the signals.
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, signal, label="Original Signal")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label="Noisy Signal")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, label="Filtered Signal")
plt.legend()

plt.tight_layout()
plt.show()
