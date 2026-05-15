import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# --- System Parameters ---
fs = 250e6               # Sampling frequency (125 MHz)
N = 8192                 # Number of samples for FFT
duration = N / fs        # Total time duration of the capture

# --- Frequency Parameters ---
# Bandwidth covers 0 to fs/2 (Nyquist). 
# We'll place two test tones arbitrarily within the bandwidth.
f1 = 20e6                # Tone 1 at 20 MHz
f2 = 45e6                # Tone 2 at 45 MHz

# --- Generate Time-Domain Signal ---
t = np.linspace(0, duration, N, endpoint=False)
# Two sine waves (amplitudes 1.0) mixed together
signal = 1.0 * np.sin(2 * np.pi * f1 * t) + 1.0 * np.sin(2 * np.pi * f2 * t)

# --- Add Noise (Optional but realistic for RF simulations) ---
noise = np.random.normal(0, 0.5, N)
signal += noise

# --- Compute Spectrum (FFT) ---
# Calculate the Fast Fourier Transform and its magnitude in dB
fft_signal = np.fft.fft(signal)
magnitude = 20 * np.log10(np.abs(fft_signal) / N)

# Get the corresponding frequency bins
freqs = np.fft.fftfreq(N, d=1/fs)

# Shift zero-frequency to the center if you want a baseband representation, 
# or just plot positive frequencies for real-valued signals.
# For simplicity, we'll plot the positive half (up to Nyquist: 62.5 MHz).
freqs_positive = freqs / 1e6 # in MHz
magnitude_positive = magnitude

# --- Plot the RF Spectrum ---
plt.figure(figsize=(10, 6))
plt.plot(freqs_positive, magnitude_positive, color='blue', linewidth=1.2)
plt.title('Simulated RF Spectrum (Two Tones)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dBFS)')
plt.xlim(0, fs / 2e6) # Limit x-axis to Nyquist
plt.ylim(-60, 10)     # Set appropriate dB scale
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fftw125-2.png')

# 1. Reshape signal to match polyphase branches
# x is input, num_channels (M) is the decimation factor
x_reshaped = magnitude_positive[:len(magnitude_positive)//8 * 8].reshape(-1, 8)
    
# 2. Reshape taps into PFB sub-filters

num_channels = 8  # Number of channels (M)
num_taps_per_branch = 12  # Number of coefficients per sub-filter
total_taps = num_channels * num_taps_per_branch

# Design a low-pass filter using a windowed sinc function
# Cutoff frequency is typically 1 / num_channels
taps = signal.firwin(total_taps, 1.0/num_channels)
p_taps = taps.reshape(8, -1)
    
# 3. Filter each branch and apply FFT
pfb_output = np.zeros((x_reshaped.shape[0], 8), dtype=complex)
for i in range(8):
    pfb_output[:, i] = signal.lfilter(p_taps[i], 1, x_reshaped[:, i])
  
# Efficiently shift all channels to baseband
pfb_ifft = np.fft.ifft(pfb_output, axis=1)

