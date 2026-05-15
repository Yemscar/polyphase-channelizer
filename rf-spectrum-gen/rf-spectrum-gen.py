import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
fs = 100e6           # Sample rate (100 MHz)
bandwidth = 10e6      # Desired signal bandwidth (10 MHz)
duration = 1e-6       # 1 microsecond
t = np.arange(0, duration, 1/fs)

# 1. Generate wideband white noise
noise = np.random.normal(0, 1, len(t)) + 1j * np.random.normal(0, 1, len(t))

# 2. Design a low-pass filter to limit bandwidth
# Cutoff at bandwidth/2 because it's a complex (IQ) baseband signal
nyquist = fs / 2
cutoff = (bandwidth / 2) / nyquist
b, a = signal.butter(5, cutoff, btype='low')

# 3. Apply filter to create band-limited signal
rf_signal = signal.filtfilt(b, a, noise)

# Visualization
plt.psd(rf_signal, Fs=fs/1e6)
plt.title(f"Simulated {bandwidth/1e6} MHz Bandwidth Signal")
plt.xlabel("Frequency (MHz)")
plt.show()
plt.savefig('rf-spectrum.png')
