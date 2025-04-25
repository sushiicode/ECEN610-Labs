import numpy as np
import matplotlib.pyplot as plt
sampling_rate = 500e6
full_scale_voltage = 1.0
adc_resolution = 13
num_stages = 6
input_frequency = 200e6

def ideal_mdac(input_signal, resolution=2.5, full_scale_voltage=1.0):
    quantization_step = full_scale_voltage / (2 ** resolution)
    quantized_signal = np.round(input_signal / quantization_step) * quantization_step
    return quantized_signal
time_array = np.arange(0, 1e-6, 1 / sampling_rate)
input_signal = np.sin(2 * np.pi * input_frequency * time_array)
processed_signal = input_signal
for _ in range(num_stages):
    processed_signal = ideal_mdac(processed_signal)

def compute_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)
ideal_signal = np.sin(2 * np.pi * input_frequency * time_array)
quantization_noise = processed_signal - ideal_signal
signal_to_noise_ratio = compute_snr(ideal_signal, quantization_noise)
print(f"Signal-to-Noise Ratio (SNR): {signal_to_noise_ratio} dB")
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_array[:1000], input_signal[:1000], label='Input Tone (200 MHz)')
plt.title('Input Tone (200 MHz)')
plt.xlabel('Time in s')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.plot(time_array[:1000], processed_signal[:1000], label='Processed Signal After MDAC Stages', color='r')
plt.title('Processed Signal After MDAC Stages')
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
