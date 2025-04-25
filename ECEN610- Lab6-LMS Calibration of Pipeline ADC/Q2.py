import numpy as np
import matplotlib.pyplot as plt
sampling_rate = 500e6
full_scale_voltage = 1.0
bit_resolution = 13
stage_count = 6
signal_frequency = 200e6
error_factors = {
    'gain_reduction': 1.01,
    'offset_error': 0.05,
    'cap_mismatch': 0.02,
    'comp_offset': 0.001,
    'amp_gain_reduction': 0.99,
    'bandwidth_limit': 0.9,
}

def mdac_with_errors(input_data, resolution=2.5, full_scale_voltage=1.0, errors=None):
    if errors is None:
        errors = error_factors
    gain_factor = errors['gain_reduction']
    input_data = input_data * gain_factor
    offset_factor = errors['offset_error']
    input_data += offset_factor * full_scale_voltage
    capacitor_mismatch_factor = errors['cap_mismatch']
    input_data *= (1 + capacitor_mismatch_factor)
    comparator_offset_factor = errors['comp_offset']
    input_data += comparator_offset_factor * full_scale_voltage
    op_amp_gain_factor = errors['amp_gain_reduction']
    input_data = np.sign(input_data) * np.abs(input_data) ** op_amp_gain_factor
    bandwidth_effect_factor = errors['bandwidth_limit']
    input_data = input_data * bandwidth_effect_factor
    quant_step_size = full_scale_voltage / (2 ** resolution)
    quantized_output = np.round(input_data / quant_step_size) * quant_step_size

    return quantized_output

time_array = np.arange(0, 1e-6, 1 / sampling_rate)
input_data = np.sin(2 * np.pi * signal_frequency * time_array)

processed_signal = input_data
for _ in range(stage_count):
    processed_signal = mdac_with_errors(processed_signal)

def calc_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

ideal_signal = np.sin(2 * np.pi * signal_frequency * time_array)
quantization_noise = processed_signal - ideal_signal

snr_result = calc_snr(ideal_signal, quantization_noise)
print(f"SNR with errors: {snr_result} dB")
def to_db(signal):
    return 20 * np.log10(np.maximum(np.abs(signal), 1e-12))
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_array[:1000], to_db(input_data[:1000]), label='Input Tone (200 MHz)')
plt.title('Input Tone (200 MHz) in dB')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [dB]')
plt.subplot(2, 1, 2)
plt.plot(time_array[:1000], to_db(processed_signal[:1000]), label='Processed Signal with Errors', color='r')
plt.title('Processed Signal with Static Errors in dB')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [dB]')
plt.tight_layout()
plt.show()
