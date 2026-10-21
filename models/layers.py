import torch
import torch.nn as nn

class FFTFrequencyLayer(nn.Module):
    """
    Computes shifted 2D Fast Fourier Transform magnitude spectra
    to capture high-frequency periodic artifacts left by GAN upsamplers.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (B, C, H, W)
        fft_complex = torch.fft.fft2(x, norm="ortho")
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        # Log-transformed magnitude spectrum
        magnitude_spectrum = torch.log(torch.abs(fft_shifted) + 1e-8)
        return magnitude_spectrum
