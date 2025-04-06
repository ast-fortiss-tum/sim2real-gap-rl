# fft_filter.py
import torch
import torch.nn as nn

class LearnableFFTFilter0(nn.Module):
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
    
class FixedFFTFilter0(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length
        # Initialize a fixed mask parameter. Using register_buffer ensures
        # that it is not considered a learnable parameter.
        self.register_buffer("mask_param", torch.zeros(signal_length))
    
    def forward(self, x):
        # x is expected to be a 1D tensor of length signal_length.
        # Compute the FFT (complex tensor).
        X = torch.fft.fft(x)
        # Apply the sigmoid on the fixed mask (resulting in 0.5 everywhere if initialized at zero).
        mask = torch.sigmoid(self.mask_param)
        # Multiply frequency coefficients elementwise.
        X_filtered = X * mask.to(x.dtype)
        # Compute inverse FFT and return only the real part.
        x_filtered = torch.fft.ifft(X_filtered)
        return x_filtered.real

class LearnableFFTFilter(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length
        # Create a learnable parameter that will serve as a frequency-domain mask.
        # Initialized to zeros so that after sigmoid, each element starts at 0.5.
        self.mask_param = nn.Parameter(torch.zeros(signal_length))
    
    def forward(self, x):
        """
        x: a 1D tensor of shape [L] (L may not equal self.signal_length)
        """
        L = x.shape[0]
        # Adjust the mask to the input length:
        if L < self.signal_length:
            # Crop the fixed mask to the first L elements.
            mask = torch.sigmoid(self.mask_param[:L])
        elif L > self.signal_length:
            # Pad the mask with ones (i.e., no attenuation) for the extra elements.
            pad = torch.ones(L - self.signal_length, device=x.device, dtype=x.dtype)
            mask = torch.cat([torch.sigmoid(self.mask_param), pad], dim=0)
        else:
            mask = torch.sigmoid(self.mask_param)
        
        # Compute FFT of the input signal.
        X = torch.fft.fft(x)
        # Apply the mask elementwise.
        X_filtered = X * mask.to(x.dtype)
        # Compute inverse FFT and return only the real part.
        x_filtered = torch.fft.ifft(X_filtered)
        return x_filtered.real

class FixedFFTFilter(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.signal_length = signal_length
        # Register a fixed mask parameter (initialized to zeros).
        # Since it is a buffer, it will not be updated during training.
        self.register_buffer("mask_param", torch.zeros(signal_length))
    
    def forward(self, x):
        """
        x: a 1D tensor of shape [L] (L may not equal self.signal_length)
        Returns:
            The denoised signal computed by applying a fixed FFT mask.
        """
        L = x.shape[0]
        # Adjust the fixed mask to match the input length:
        if L < self.signal_length:
            mask = torch.sigmoid(self.mask_param[:L])
        elif L > self.signal_length:
            pad = torch.ones(L - self.signal_length, device=x.device, dtype=x.dtype)
            mask = torch.cat([torch.sigmoid(self.mask_param), pad], dim=0)
        else:
            mask = torch.sigmoid(self.mask_param)
        
        # Compute FFT of the input signal.
        X = torch.fft.fft(x)
        # Apply the mask elementwise.
        X_filtered = X * mask.to(x.dtype)
        # Compute inverse FFT and return only the real part.
        x_filtered = torch.fft.ifft(X_filtered)
        return x_filtered.real
