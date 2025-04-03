import numpy as np
import matplotlib.pyplot as plt

def adaptive_fft_filter(signal, energy_threshold=0.95):
    """
    Apply an adaptive FFT low-pass filter by selecting a cutoff frequency 
    that preserves a specified percentage of the total spectral energy.
    
    Parameters:
      signal: 1D numpy array of the noisy signal.
      energy_threshold: Fraction of total energy to retain (default 95%).
    
    Returns:
      filtered_signal: The denoised signal.
      cutoff: The frequency (normalized index) used as cutoff.
    """
    N = len(signal)
    # Compute FFT and corresponding power spectrum
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N)
    energy = np.abs(X)**2

    # Sort frequencies by energy in descending order
    sorted_idx = np.argsort(energy)[::-1]
    sorted_energy = energy[sorted_idx]
    cum_energy = np.cumsum(sorted_energy)
    total_energy = cum_energy[-1]
    
    # Find the index where cumulative energy reaches the threshold
    cutoff_idx = sorted_idx[np.where(cum_energy >= energy_threshold * total_energy)[0][0]]
    cutoff = np.abs(freqs[cutoff_idx])
    
    # Create a symmetric mask for frequencies below cutoff
    mask = np.abs(freqs) <= cutoff
    X_filtered = X * mask
    
    filtered_signal = np.fft.ifft(X_filtered).real
    return filtered_signal, cutoff

# Example usage:
N = 256
t = np.linspace(0, 1, N)
# A low-frequency sine wave as the clean signal.
signal = np.sin(2 * np.pi * 5 * t)
# Add high-frequency noise.
noisy_signal = signal + 0.5 * np.random.randn(N)

filtered_signal, cutoff = adaptive_fft_filter(noisy_signal, energy_threshold=0.75)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, signal, label="Original Signal")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label="Noisy Signal")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, label="Adaptive FFT Filtered Signal\nCutoff: {:.3f}".format(cutoff))
plt.legend()
plt.tight_layout()
plt.show()
