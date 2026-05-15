import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
fs = 125e6               # Sampling frequency (125 MHz)
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
freqs_positive = freqs[:N//2] / 1e6 # in MHz
magnitude_positive = magnitude[:N//2]

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
plt.savefig('fftw125.png')
