# fft_filter.py
import torch
import torch.nn as nn

class LearnableFFTFilter(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length
        # Create a learnable parameter that will serve as a frequency–domain mask.
        # We parameterize in raw form and then use a sigmoid to constrain values between 0 and 1.
        self.mask_param = nn.Parameter(torch.zeros(signal_length))
    
    def forward(self, x):
        # x is expected to be a 1D tensor of length signal_length.
        # Compute the FFT (returns a complex tensor).
        X = torch.fft.fft(x)
        # Apply a sigmoid to obtain a mask between 0 and 1.
        mask = torch.sigmoid(self.mask_param)
        # Multiply elementwise (mask is cast to the complex type).
        X_filtered = X * mask.to(x.dtype)
        # Compute the inverse FFT and return only the real part.
        x_filtered = torch.fft.ifft(X_filtered)
        return x_filtered.real
