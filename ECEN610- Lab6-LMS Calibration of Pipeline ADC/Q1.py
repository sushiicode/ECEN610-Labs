import numpy as np
import matplotlib.pyplot as plt

# Constants
sampling_rate = 500e6  # Sampling frequency 500 MHz
full_scale_voltage = 1.0  # Full-scale voltage (1V)
adc_resolution = 13  # ADC resolution in bits
num_stages = 6  # Number of stages (MDAC stages)
input_frequency = 200e6  # Input tone frequency (200 MHz)

# MDAC Transfer Function (ideal 2.5-bit resolution)
def ideal_mdac(input_signal, resolution=2.5, full_scale_voltage=1.0):
    # Quantization step size (assuming ideal MDAC)
    quantization_step = full_scale_voltage / (2 ** resolution)
    
    # Quantize the signal to the 2.5-bit resolution
    quantized_signal = np.round(input_signal / quantization_step) * quantization_step
    
    return quantized_signal

# Generate input tone (200 MHz) with ideal signal
time_array = np.arange(0, 1e-6, 1 / sampling_rate)  # 1 microsecond of signal
input_signal = np.sin(2 * np.pi * input_frequency * time_array)

# Pass the signal through the MDAC stages
processed_signal = input_signal
for _ in range(num_stages):
    processed_signal = ideal_mdac(processed_signal)

# SNR Calculation
def compute_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# Compute noise as the difference between ideal and quantized output
ideal_signal = np.sin(2 * np.pi * input_frequency * time_array)  # Ideal signal (no quantization)
quantization_noise = processed_signal - ideal_signal

# Calculate SNR
signal_to_noise_ratio = compute_snr(ideal_signal, quantization_noise)
print(f"Signal-to-Noise Ratio (SNR): {signal_to_noise_ratio} dB")

# Plot results
plt.figure(figsize=(10, 6))

# Plot input tone
plt.subplot(2, 1, 1)
plt.plot(time_array[:1000], input_signal[:1000], label='Input Tone (200 MHz)')
plt.title('Input Tone (200 MHz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot output signal after MDAC stages
plt.subplot(2, 1, 2)
plt.plot(time_array[:1000], processed_signal[:1000], label='Processed Signal After MDAC Stages', color='r')
plt.title('Processed Signal After MDAC Stages')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Adjust layout and show plots
plt.tight_layout()
plt.show()
