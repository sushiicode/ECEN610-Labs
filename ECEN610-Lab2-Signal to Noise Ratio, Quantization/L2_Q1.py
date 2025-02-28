#Question-1

#Q1.A

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

f_sig = 2e6
pd = 1 / f_sig
tv = np.linspace(0, 100 * pd, 10000)
org_wave = np.sin(2 * np.pi * f_sig * tv)

fig1, org_plot = plt.subplots(figsize=(8, 4))
org_plot.plot(tv * 1e6, org_wave, 'g', label="Original Signal")
org_plot.set_xlabel("Time in μs")
org_plot.set_ylabel("Amplitude of Signal")
org_plot.set_title("Original Signal Plot")
org_plot.grid()
org_plot.legend(loc='upper right')

f_Samp = 5e6
samp_pd = 1 / f_Samp
samp_tvector = np.arange(0, 100 * pd, samp_pd)
wave_sampled = np.sin(2 * np.pi * f_sig * samp_tvector)

fig2, samp_plt = plt.subplots(figsize=(8, 4))
samp_plt.plot(tv * 1e6, org_wave, 'g', alpha=0.5, label="Original Signal Plot")
samp_plt.plot(samp_tvector * 1e6, wave_sampled, 'b', markersize=4, label="Sampled Elements")
samp_plt.set_xlabel("Time in μs")
samp_plt.set_ylabel("Amplitude of Signal")
samp_plt.set_title("Sampled Signal (5 MHz)")
samp_plt.grid()
samp_plt.legend(loc='upper right')

plt.show()

samp_pwr_avg = np.mean(pow(wave_sampled, 2))
print("Average power (sampled signal):", samp_pwr_avg)

sr_in_db = 50
print("Desired SNR in dB:", sr_in_db)
snr_linear = pow(10, (sr_in_db / 10))
print("SNR- Linear Scale:", snr_linear)

noise_pwr_avg = samp_pwr_avg / snr_linear
noise_var = noise_pwr_avg
print("Calculated noise variance:", noise_var)

noise_mean = 0
noise_gaussian = np.random.normal(noise_mean, np.sqrt(noise_var), len(samp_tvector))

noise_pwr_avg_org = np.mean(pow(noise_gaussian, 2))
print("Measured power (noise signal):", noise_pwr_avg_org)

fig3, plt_Noise = plt.subplots(figsize=(8, 4))
plt_Noise.plot(samp_tvector * 1e6, noise_gaussian, 'g', label="Gaussian Noise")
plt_Noise.set_xlabel("Time in μs")
plt_Noise.set_ylabel("Amplitude of Signal")
plt_Noise.set_title("Gaussian Noise Signal Plot")
plt_Noise.grid()
plt_Noise.legend(loc='upper right')

noise_sig_sampled = wave_sampled + noise_gaussian

fig4, noise_sig_plt = plt.subplots(figsize=(8, 4))
noise_sig_plt.plot(samp_tvector * 1e6, noise_sig_sampled, 'g', label="Noise Sampled Signal Plot")
noise_sig_plt.set_xlabel("Time in μs")
noise_sig_plt.set_ylabel("Amplitude of Signal")
noise_sig_plt.set_title("Noise Sampled Signal Plot")
noise_sig_plt.grid()
noise_sig_plt.legend(loc='upper right')

f_values, psd_value = signal.periodogram(noise_sig_sampled, f_Samp, nfft=230)

fig5, plot_psd = plt.subplots(figsize=(8, 4))
plot_psd.plot(f_values, psd_value, 'g', label="Power Spectral Density Plot")
plot_psd.set_xlabel("Frequency in Hz")
plot_psd.set_ylabel("Power Spectral Density")
plot_psd.set_title("Power Spectral Density (Noise Sampled Signal)")
plot_psd.grid()
plot_psd.legend(loc='upper right')

frequency_peak = f_values[np.argmax(psd_value)]

sig_pwr = 0
noise_pwr = 0

for i in range(len(f_values)):
    if f_values[i] == frequency_peak:
        sig_pwr += psd_value[i]
    else:
        noise_pwr += psd_value[i]

print("Measured signal power from PSD:", sig_pwr)
print("Measured noise power from PSD:", noise_pwr)

actual_snr = 10 * np.log10(sig_pwr / noise_pwr)
print(f"SNR Which is computed from PSD of noise sampled signal: {actual_snr:.2f} dB")

plt.show()


#Q1.B

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


freq_sig = 2e6
pd = 1 / freq_sig
tvect = np.linspace(0, 100 * pd, 10000)
org_wave = np.sin(2 * np.pi * freq_sig * tvect)


fig1, org_plot = plt.subplots(figsize=(8, 4))
org_plot.plot(tvect * 1e6, org_wave, 'g', label="Original Signal Plot")
org_plot.set_xlabel("Time in µs")
org_plot.set_ylabel("Amplitude of Signal")
org_plot.set_title("Original Signal Plot")
org_plot.grid()
org_plot.legend(loc='upper right')


freq_samp = 5e6
samp_pd = 1 / freq_samp
samp_tvect = np.arange(0, 100 * pd, samp_pd)
wave_sampled = np.sin(2 * np.pi * freq_sig * samp_tvect)


windows = {
    "Hanning_Window": np.hanning(len(wave_sampled)),
    "Hamming_Window": np.hamming(len(wave_sampled)),
    "Blackman_Window": np.blackman(len(wave_sampled))
}

wind_psd = {}
window_sig = {}

for window_name, window in windows.items():
    windowed_signal = wave_sampled * window
    window_sig[window_name] = windowed_signal
    f_value, psd_values = signal.periodogram(windowed_signal, freq_samp, nfft=230)
    wind_psd[window_name] = (f_value, psd_values)


for window_name, windowed_signal in window_sig.items():
    fig, m = plt.subplots(figsize=(8, 4))
    m.plot(samp_tvect * 1e6, windowed_signal, 'g', label=f"{window_name} plot")
    m.set_xlabel("Time in µs")
    m.set_ylabel("Amplitude of Signal")
    m.set_title(f"{window_name}")
    m.grid()
    m.legend(loc='upper right')


for window_name, (f_value, psd_values) in wind_psd.items():
    fig, m = plt.subplots(figsize=(8, 4))
    m.plot(f_value, psd_values, 'g', label=f"{window_name}")
    m.set_xlabel("Frequency in Hz")
    m.set_ylabel("Power Spectral Density (PSD)")
    m.set_title(f"Power Spectral Density Plot- {window_name}")
    m.grid()
    m.legend(loc='upper right')

plt.show()

