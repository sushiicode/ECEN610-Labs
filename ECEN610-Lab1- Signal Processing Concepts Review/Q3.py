#Q3

#Q3.A

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

sig_freq = 2e6
f_samp = 5e6
samp_period = 1 / f_samp
time = np.linspace(0, 10 * (1 / sig_freq), 10000)
time_samp = np.arange(0, 10 * (1 / sig_freq), samp_period)
org_sig = np.cos(2 * np.pi * sig_freq * time)
Samp_Signal = np.cos(2 * np.pi * sig_freq * time_samp)
Sig_dft = fft(Samp_Signal, 64)
Mag_dft = np.abs(Sig_dft)

fig, (m1, m2, m3) = plt.subplots(3, 1, figsize=(10, 8))

m1.plot(time, org_sig, color='red')
m1.set_title('Original Signal: $x(t) = \cos(2 \pi f t)$')
m1.set_xlabel('Time [s]')
m1.set_ylabel('Amplitude')

m2.plot(time_samp, Samp_Signal, color='blue')
m2.set_title('Sampled Signal: $x_s[n]$')
m2.set_xlabel('Time [s]')
m2.set_ylabel('Amplitude')

frequency_axis = np.fft.fftfreq(64, samp_period)
m3.plot(frequency_axis, Mag_dft, color='green')
m3.set_title('Magnitude of the 64-point DFT')
m3.set_xlabel('Frequency [Hz]')
m3.set_ylabel('Magnitude')

plt.tight_layout()
plt.show()



#Q3. B

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

freq1 = 200e6
freq2 = 400e6
samp_freq = 1e9
org_time = np.linspace(0, 10 * (1 / (freq1 + freq2)), 10000)
samp_time = np.arange(0, 10 * (1 / (freq1 + freq2)), 1 / samp_freq)
org_sig = np.cos(2 * np.pi * freq1 * org_time) + np.cos(2 * np.pi * freq2 * org_time)
samp_sig = np.cos(2 * np.pi * freq1 * samp_time) + np.cos(2 * np.pi * freq2 * samp_time)
sig_dft = fft(samp_sig, 64)
mag_dft = np.abs(sig_dft)
freqz = np.fft.fftfreq(64, 1/samp_freq)
pos_freqz = freqz[:32]
pos_mag_dft = mag_dft[:32]

fig, (m1, m2, m3) = plt.subplots(3, 1, figsize=(10, 8))

m1.plot(org_time, org_sig, color='red')
m1.set_title('Original Signal: $y(t) = \cos(2\pi F_1 t) + \cos(2\pi F_2 t)$')
m1.set_xlabel('Time [s]')
m1.set_ylabel('Amplitude')

m2.stem(samp_time, samp_sig, use_line_collection=True, linefmt='r', markerfmt='ro')
m2.set_title('Sampled Signal at $F_s = 1$ GHz')
m2.set_xlabel('Time [s]')
m2.set_ylabel('Amplitude')

m3.plot(pos_freqz, pos_mag_dft, color='red')
m3.set_title('Magnitude of 64-point DFT (Positive Frequencies)')
m3.set_xlabel('Frequency [Hz]')
m3.set_ylabel('Magnitude')

plt.tight_layout()
plt.show()


#Q3.C

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

freq1 = 200e6
freq2 = 400e6
f_samp = 500e6
org_time = np.linspace(0, 10 * (1 / (freq1 + freq2)), 10000)
samp_time = np.arange(0, 10 * (1 / (freq1 + freq2)), 1 / f_samp)
org_sig = np.cos(2 * np.pi * freq1 * org_time) + np.cos(2 * np.pi * freq2 * org_time)
sig_sampled = np.cos(2 * np.pi * freq1 * samp_time) + np.cos(2 * np.pi * freq2 * samp_time)
sig_dft = fft(sig_sampled, 64)
mag_dft = np.abs(sig_dft)
freqz = np.fft.fftfreq(64, 1/f_samp)
pos_freqz = freqz[:32]
pos_mag_dft = mag_dft[:32]

fig, (m1, ax1, m3) = plt.subplots(3, 1, figsize=(10, 8))

m1.plot(org_time, org_sig, color='red')
m1.set_title('Original Signal: $y(t) = \cos(2\pi F_1 t) + \cos(2\pi F_2 t)$')
m1.set_xlabel('Time [s]')
m1.set_ylabel('Amplitude')

ax1.stem(samp_time, sig_sampled, use_line_collection=True, linefmt='r', markerfmt='ro')
ax1.set_title('Sampled Signal at $F_s = 500$ MHz')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude')

m3.plot(pos_freqz, pos_mag_dft, color='red')
m3.set_title('Magnitude of 64-point DFT (Positive Frequencies)')
m3.set_xlabel('Frequency [Hz]')
m3.set_ylabel('Magnitude')

plt.tight_layout()
plt.show()
