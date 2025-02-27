#Q4


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift

freq1 = 2e6
samp_freq = 5e6
org_time = np.linspace(0, 10 * (1 / freq1), 10000)
samp_time = np.arange(0, 10 * (1 / freq1), 1 / samp_freq)
signal_org = np.cos(2 * np.pi * freq1 * org_time)
sig_sampled = np.cos(2 * np.pi * freq1 * samp_time)
sig_dft = fft(sig_sampled, 50)
mag_dft = np.abs(sig_dft)
black_win = np.blackman(len(mag_dft))
dft_wind = mag_dft * black_win
freqz = np.fft.fftfreq(len(mag_dft), 1/samp_freq)
freqz = fftshift(freqz)
resp_in_db = 20 * np.log10(np.abs(fftshift(dft_wind / abs(dft_wind).max())))

fig, (m1, m2, m3, m4) = plt.subplots(4, 1, figsize=(10, 8))

m1.plot(org_time, signal_org, color='red')
m1.set_title('Original Signal: $x(t) = \cos(2\pi f_1 t)$')
m1.set_xlabel('Time [s]')
m1.set_ylabel('Amplitude')

m2.plot(samp_time, sig_sampled, color='red')
m2.set_title('Sampled Signal at $F_s = 5$ MHz')
m2.set_xlabel('Time [s]')
m2.set_ylabel('Amplitude')

m3.plot(freqz, resp_in_db, color='red')
m3.set_title('Blackman Windowed Frequency Response')
m3.set_xlabel('Frequency [Hz]')
m3.set_ylabel('Magnitude [dB]')

m4.plot(freqz, mag_dft, color='red')
m4.set_title('Magnitude of the DFT of Sampled Signal')
m4.set_xlabel('Frequency [Hz]')
m4.set_ylabel('Magnitude')

plt.tight_layout()
plt.show()
