import numpy as np
import matplotlib.pyplot as plt

V_max = 1.0
bit_depth = 13
num_stages = 6
bits_per_stage = 2
sample_rate = 500e6
tone_frequency = 200e6
sim_duration = 1e-6
num_samples = int(sample_rate * sim_duration)

calibration_stages = 4
learning_rate = 0.01
max_iter = 500
small_constant = 1e-6

time_array = np.linspace(0, sim_duration, num_samples, endpoint=False)
clean_signal = 0.5 * np.sin(2 * np.pi * tone_frequency * time_array)
signal_rms = np.sqrt(np.mean(clean_signal ** 2))
noise_rms = signal_rms / np.sqrt(10 ** (80 / 10))
noisy_signal = clean_signal + np.random.normal(0, noise_rms, num_samples)

def ideal_adc(input_signal, bit_depth=16, V_max=1.0):
    levels = 2 ** bit_depth
    lsb = V_max / levels
    digital_output = np.round((input_signal + V_max / 2) / lsb)
    digital_output = np.clip(digital_output, 0, levels - 1)
    output_signal = (digital_output * lsb) - (V_max / 2)
    return digital_output, output_signal

def mdac_stage(input_signal, V_max=1.0, gain_error=0.01, noise_rms=1e-3, gain_corrections=None):
    thresholds = np.array([-0.25, -0.125, 0, 0.125, 0.25])
    levels = np.array([-0.5, -0.25, 0, 0.25, 0.5, 0.5])

    indices = np.digitize(input_signal, thresholds, right=True)
    output_dac = levels[indices]

    G_ideal = 2 ** bits_per_stage
    residue_ideal = (input_signal - output_dac) * G_ideal

    A_OL = 1000
    GBW_factor = 0.8
    G_effective = G_ideal * GBW_factor
    V_in = residue_ideal / G_ideal
    V_out = (A_OL * V_in +
             0.1 * A_OL * V_in**2 +
             0.2 * A_OL * V_in**3 +
             0.15 * A_OL * V_in**4 +
             0.1 * A_OL * V_in**5) / A_OL
    residue = G_effective * V_out

    if gain_corrections is not None:
        correction = (gain_corrections[0] * residue +
                      gain_corrections[1] * residue**2 +
                      gain_corrections[2] * residue**3 +
                      gain_corrections[3] * residue**4 +
                      gain_corrections[4] * residue**5)
        residue = correction / np.max(np.abs(correction)) * np.max(np.abs(residue))

    residue += np.random.normal(0, noise_rms, size=input_signal.shape)
    residue = np.clip(residue, -V_max / 2, V_max / 2)

    return indices, residue

def pipeline_adc(input_signal, num_stages, bits_per_stage, V_max, gain_corrections_per_stage):
    digital_outputs = []
    residue = input_signal.copy()
    stage_residues = []

    for stage in range(num_stages):
        gain_error = 0.01 * (1 + 0.1 * stage)
        corrections = gain_corrections_per_stage[stage] if stage < len(gain_corrections_per_stage) else None
        codes, residue = mdac_stage(residue, V_max, gain_error, 1e-3, corrections)
        digital_outputs.append(codes)
        stage_residues.append(residue.copy())

    return digital_outputs, stage_residues

def calculate_snr(signal, sample_rate):
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

decimation_factors = [10, 100, 1000, 10000]
quantization_limit = 6.02 * bit_depth + 1.76

ref_digital_output, ref_output_signal = ideal_adc(noisy_signal, bit_depth=16, V_max=V_max)

plt.figure(figsize=(12, 12))

for dec_idx, dec_factor in enumerate(decimation_factors):
    gain_corrections_per_stage = [np.array([1.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(calibration_stages)]
    error_values = []
    snr_values = []
    weights_history = [[[] for _ in range(5)] for _ in range(calibration_stages)]

    for iteration in range(max_iter):
        digital_outputs, stage_residues = pipeline_adc(noisy_signal, num_stages, bits_per_stage, V_max, gain_corrections_per_stage)

        Dout = np.zeros(num_samples, dtype=int)
        for stage in range(num_stages):
            Dout += digital_outputs[stage] * (2 ** (bits_per_stage * (num_stages - stage - 1)))

        output_signal = (Dout / (2 ** bit_depth)) * V_max - (V_max / 2)
        error = ref_output_signal - output_signal

        if iteration % dec_factor == 0:
            error_decimated = error[::dec_factor]
            for stage in range(calibration_stages):
                prev_residue = noisy_signal if stage == 0 else stage_residues[stage - 1]
                prev_residue_decimated = prev_residue[::dec_factor]
                signal_power = np.mean(prev_residue_decimated ** 2)
                for order in range(5):
                    gain_corrections_per_stage[stage][order] += (learning_rate * np.mean(error_decimated * prev_residue_decimated ** (order + 1)) /
                                                                 (signal_power + small_constant))
                    weights_history[stage][order].append(gain_corrections_per_stage[stage][order])

        error_values.append(np.mean(error ** 2))
        snr_value = calculate_snr(output_signal, sample_rate)
        snr_values.append(snr_value)

        if snr_value >= quantization_limit - 1:
            break

    plt.subplot(len(decimation_factors), 3, dec_idx * 3 + 1)
    plt.plot(error_values)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title(f"Error (Decimation = {dec_factor})")
    plt.grid(True)

    plt.subplot(len(decimation_factors), 3, dec_idx * 3 + 2)
    for stage in range(calibration_stages):
        plt.plot(np.array(weights_history[stage]).T)
    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title(f"Weight Convergence (Decimation = {dec_factor})")
    plt.grid(True)

    plt.subplot(len(decimation_factors), 3, dec_idx * 3 + 3)
    plt.plot(snr_values)
    plt.xlabel("Iteration")
    plt.ylabel("SNR (dB)")
    plt.title(f"SNR Convergence (Decimation = {dec_factor})")
    plt.grid(True)

plt.tight_layout()
plt.show()
