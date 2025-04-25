import numpy as np
import matplotlib.pyplot as plt

V_full_scale = 1.0
num_bits = 13
num_stages = 6
bits_per_stage = 2
sampling_freq = 500e6
tone_freq = 200e6
simulation_time = 1e-6
num_samples = int(sampling_freq * simulation_time)

calibration_stages = 4
step_size = 0.01
max_iterations = 500
small_constant = 1e-6

time_vector = np.linspace(0, simulation_time, num_samples, endpoint=False)
input_signal = 0.5 * np.sin(2 * np.pi * tone_freq * time_vector)

def reference_adc(input_signal, num_bits=16, V_full_scale=1.0):
    levels = 2 ** num_bits
    lsb = V_full_scale / levels
    digital_output = np.round((input_signal + V_full_scale / 2) / lsb)
    digital_output = np.clip(digital_output, 0, levels - 1)
    output_signal = (digital_output * lsb) - (V_full_scale / 2)
    return digital_output, output_signal

def mdac_stage(input_signal, V_full_scale=1.0, gain_error=0.01, noise_rms=1e-3, gain_correction=1.0):
    thresholds = np.array([-0.25, -0.125, 0, 0.125, 0.25])
    levels = np.array([-0.5, -0.25, 0, 0.25, 0.5, 0.5])

    indices = np.digitize(input_signal, thresholds, right=True)
    output_dac = levels[indices]

    G_ideal = 2 ** bits_per_stage
    G = G_ideal * (1 + gain_error) * gain_correction

    residue = (input_signal - output_dac) * G
    residue += np.random.normal(0, noise_rms, size=input_signal.shape)
    residue = np.clip(residue, -V_full_scale / 2, V_full_scale / 2)

    return indices, residue

def pipeline_adc(input_signal, num_stages, bits_per_stage, V_full_scale, gain_corrections):
    digital_outputs = []
    residue = input_signal.copy()
    stage_residues = []

    for stage in range(num_stages):
        gain_error = 0.01 * (1 + 0.1 * stage)
        correction = gain_corrections[stage] if stage < len(gain_corrections) else 1.0

        codes, residue = mdac_stage(residue, V_full_scale, gain_error, 1e-3, correction)
        digital_outputs.append(codes)
        stage_residues.append(residue.copy())

    return digital_outputs, stage_residues

def calculate_snr(signal, sampling_freq):
    N = len(signal)
    fft_signal = np.fft.fft(signal * np.hanning(N))
    fft_magnitude = np.abs(fft_signal[:N // 2])
    signal_bin = np.argmax(fft_magnitude)
    signal_power = fft_magnitude[signal_bin] ** 2
    noise_bins = np.ones(len(fft_magnitude), dtype=bool)
    noise_bins[[0, signal_bin]] = False
    noise_power = np.sum(fft_magnitude[noise_bins] ** 2)
    snr_value = 10 * np.log10(signal_power / noise_power)
    return snr_value

gain_corrections = np.ones(calibration_stages)
error_values = []
weights_history = [[] for _ in range(calibration_stages)]
snr_values = []
quantization_limit = 6.02 * num_bits + 1.76

ref_digital_output, ref_output_signal = reference_adc(input_signal, num_bits=16, V_full_scale=V_full_scale)

for iteration in range(max_iterations):
    digital_outputs, stage_residues = pipeline_adc(input_signal, num_stages, bits_per_stage, V_full_scale, gain_corrections)

    Dout = np.zeros(num_samples, dtype=int)
    for stage in range(num_stages):
        Dout += digital_outputs[stage] * (2 ** (bits_per_stage * (num_stages - stage - 1)))

    output_signal = (Dout / (2 ** num_bits)) * V_full_scale - (V_full_scale / 2)
    error = ref_output_signal - output_signal
    error_values.append(np.mean(error ** 2))

    for stage in range(calibration_stages):
        prev_residue = input_signal if stage == 0 else stage_residues[stage - 1]
        signal_power = np.mean(prev_residue ** 2)
        gain_corrections[stage] += step_size * np.mean(error * prev_residue) / (signal_power + small_constant)
        weights_history[stage].append(gain_corrections[stage])

    snr_value = calculate_snr(output_signal, sampling_freq)
    snr_values.append(snr_value)

    if snr_value >= quantization_limit - 1:
        iterations_to_converge = iteration + 1
        break
else:
    iterations_to_converge = max_iterations

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(error_values, 'g')
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("NLMS Error Convergence")
plt.grid(True)

plt.subplot(3, 1, 2)
for stage in range(calibration_stages):
    plt.plot(weights_history[stage], label=f"Stage {stage + 1}", color='g')
plt.xlabel("Iteration")
plt.ylabel("Weights")
plt.title("NLMS Weights Convergence")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(snr_values, 'g')
plt.xlabel("Iteration")
plt.ylabel("SNR (dB)")
plt.title("SNR Convergence")
plt.grid(True)

plt.tight_layout()
plt.savefig("nlms_convergence.png")

print(f"Number of iterations to reach quantization limit: {iterations_to_converge}")
print(f"Final weights: {gain_corrections}")
