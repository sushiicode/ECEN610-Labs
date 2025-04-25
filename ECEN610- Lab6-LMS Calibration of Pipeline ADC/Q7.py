import numpy as np
import matplotlib.pyplot as plt

Vfs = 1.0
num_bits = 13
num_tones = 128
tone_bw = 200e6
fs = 500e6
sim_time = 1e-6
num_samples = int(fs * sim_time)
max_iter = 500
step_size = 0.01
small_constant = 1e-6

def gen_multitone_signal(num_tones, tone_bw, fs, sim_time):
    t = np.linspace(0, sim_time, num_samples, endpoint=False)
    freqs = np.linspace(0, tone_bw, num_tones)
    symbols = np.random.choice([-1, 1], size=num_tones)
    signal = np.zeros(num_samples)

    for f, sym in zip(freqs, symbols):
        signal += sym * np.cos(2 * np.pi * f * t)

    return t, signal, freqs, symbols

def ideal_adc(input_signal, num_bits=16, Vfs=1.0):
    levels = 2 ** num_bits
    lsb = Vfs / levels
    digital_output = np.round((input_signal + Vfs / 2) / lsb)
    digital_output = np.clip(digital_output, 0, levels - 1)
    output_signal = (digital_output * lsb) - (Vfs / 2)
    return digital_output, output_signal

def adc_with_linear(input_signal, gain_error=0.01, offset_error=0.01):
    distorted_signal = (1 + gain_error) * input_signal + offset_error
    return distorted_signal

def adc_with_nonlinear(input_signal, nonlinear_params):
    nonlinear_signal = input_signal + nonlinear_params[0] * input_signal ** 2 + nonlinear_params[1] * input_signal ** 3
    return nonlinear_signal

def estimate_bpsk(signal, freqs, fs, num_tones):
    N = len(signal)
    fft_signal = np.fft.fft(signal * np.hanning(N))
    fft_magnitude = np.abs(fft_signal[:N // 2])
    estimated_symbols = np.sign(np.real(fft_signal[:num_tones]))
    return estimated_symbols

def calc_mse_and_ber(true_symbols, estimated_symbols):
    mse = np.mean((true_symbols - estimated_symbols) ** 2)
    ber = np.sum(true_symbols != estimated_symbols) / len(true_symbols)
    return mse, ber

def lms(input_signal, ref_signal, max_iter, decimation_factor):
    gain_corrections = np.zeros(5)

    error_vals = []
    snr_vals = []
    corrected_signal = input_signal.copy()

    for iter in range(max_iter):
        error = ref_signal - corrected_signal

        if iter % decimation_factor == 0:
            error_decimated = error[::decimation_factor]
            input_decimated = input_signal[::decimation_factor]
            signal_power = np.mean(input_decimated ** 2)

            for order in range(5):
                gain_corrections[order] += step_size * np.mean(error_decimated * input_decimated ** (order + 1)) / (
                            signal_power + small_constant)

        corrected_signal = input_signal + np.sum(
            [gain_corrections[order] * input_signal ** (order + 1) for order in range(5)], axis=0)

        snr_value = 10 * np.log10(np.mean(ref_signal ** 2) / np.mean(error ** 2))
        error_vals.append(np.mean(error ** 2))
        snr_vals.append(snr_value)

    return corrected_signal, error_vals, snr_vals, gain_corrections

def simulate():
    t, multitone_signal, freqs, true_symbols = gen_multitone_signal(num_tones, tone_bw, fs, sim_time)

    linear_signal = adc_with_linear(multitone_signal, gain_error=0.01, offset_error=0.01)

    nonlinear_params = [0.01, -0.005]
    nonlinear_signal = adc_with_nonlinear(linear_signal, nonlinear_params)

    decimation_factors = [10, 100, 1000, 10000]

    mse_vals = []
    ber_vals = []

    for decimation_factor in decimation_factors:
        corrected_signal, error_vals, snr_vals, gain_corrections = lms(nonlinear_signal, multitone_signal, max_iter,
                                                                       decimation_factor)

        estimated_symbols = estimate_bpsk(corrected_signal, freqs, fs, num_tones)

        mse, ber = calc_mse_and_ber(true_symbols, estimated_symbols)

        mse_vals.append(mse)
        ber_vals.append(ber)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(error_vals, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.title(f'MSE Convergence (Decimation={decimation_factor})')

        plt.subplot(1, 2, 2)
        plt.plot(snr_vals, color='green')
        plt.xlabel('Iterations')
        plt.ylabel('SNR (dB)')
        plt.title(f'SNR Convergence (Decimation={decimation_factor})')

        plt.tight_layout()
        plt.show()

    return mse_vals, ber_vals

mse_vals, ber_vals = simulate()
print("MSE values for different decimation factors:", mse_vals)
print("BER values for different decimation factors:", ber_vals)
