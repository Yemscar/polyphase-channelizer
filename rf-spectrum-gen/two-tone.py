import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def generate_rf_spectrum():
    # --- Parameters ---
    fs = 125e6           # Sample Rate / Bandwidth (in Hz)
    f1 = 25e6            # Frequency of Tone 1 (in Hz)
    f2 = 50e6            # Frequency of Tone 2 (in Hz)
    duration = 0.01      # Duration of the signal in seconds
    noise_power = 0.1    # Add a baseline noise floor
    
    # --- Time Vector & Signal Generation ---
    t = np.arange(0, duration, 1 / fs)
    
    # Two tones + Gaussian noise
    signal = (np.cos(2 * np.pi * f1 * t) + 
              np.cos(2 * np.pi * f2 * t) + 
              np.random.normal(0, noise_power, t.shape))

    # --- Power Spectral Density (PSD) ---
    # We use Welch's method to estimate the RF spectrum accurately
    f, psd = welch(signal, fs=fs, nperseg=2048)
    psd_db = 10 * np.log10(psd / np.max(psd)) # Normalize to dB

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(f / 1e3, psd_db, color='cyan', linewidth=1.5)
    
    plt.title('Simulated RF Spectrum (Two Tones)')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.xlim(0, fs / 1e3)
    plt.ylim(-60, 5) # Set dynamic range
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plot-two-tone.png')

if __name__ == "__main__":
    generate_rf_spectrum()
