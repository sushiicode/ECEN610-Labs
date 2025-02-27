# #Q1 Python Code
#
# #Q1.A
#
# # FIR Filter Response
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
#
# num=[0.5,0.5,0.5,0.5]
# den=[1]
# w, h = freqz(num,den)
# plt.plot(w, 20 * np.log10(abs(h)))
# plt.xscale('log')
# plt.title('Magnitude response (FIR filter)')
# plt.xlabel('Frequency in rad/sample')
# plt.ylabel('Magnitude in dB')
# plt.grid(True)
# plt.show()
#
#
# # IIR Filter Response
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
#
# a=[0.7,0.7]
# b=[1,-0.7]
#
# A, B = freqz(a,b)
# plt.plot(A, 20 * np.log10(abs(B)))
# plt.xscale('log')
# plt.title('Magnitude response of the IIR filter')
# plt.xlabel('Frequency [rad/sample]')
# plt.ylabel('Magnitude [dB]')
# plt.grid(True)
# plt.show()
#
#
# #Q1.B
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
#
# num=[1,1,1,1,1]
# den=[1]
# a, b = freqz(num,den)
# plt.plot(a, 20 * np.log10(abs(b)))
# plt.xscale('log')
# plt.title('Magnitude response of the FIR filter')
# plt.xlabel('Frequency [rad/sample]')
# plt.ylabel('Magnitude [dB]')
# plt.grid(True)
# plt.show()
#
# #Modified code for finding zeros and poles
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz, tf2zpk
#
# a = [1, 1,1,1,1]
# b = [1]
# A, B = freqz(a, b)
# plt.plot(A, 20 * np.log10(abs(B)))
# plt.xscale('log')
# plt.title('Magnitude response (FIR filters) ')
# plt.xlabel('Frequency in rad/sample')
# plt.ylabel('Magnitude in dB')
# plt.grid(True)
# plt.show()
# zeros, poles, gain = tf2zpk(a,b)
# print("Zeros ( FIR filter) :")
# print(zeros)
# print("Poles (FIR filter) :")
# print(poles)



#Q2

#Q2.A

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# sampling_rate = 500e6
# f_1 = 300e6
# f_2 = 800e6
#
# Sig1 = np.linspace(0, 10 / f_1, 15000)
# Sig2 = np.linspace(0, 10 / f_2, 15000)
#
# Time_samp_1 = np.arange(0, 10 / f_1, 1 / sampling_rate)
# Time_samp_2 = np.arange(0, 10 / f_2, 1 / sampling_rate)
#
# Sig_1 = np.cos(2 * np.pi * f_1 * Sig1)
# Sig_2 = np.cos(2 * np.pi * f_2 * Sig2)
#
# sampsig_1 = np.cos(2 * np.pi * f_1 * Time_samp_1)
# sampsig_2 = np.cos(2 * np.pi * f_2 * Time_samp_2)
#
# fig, (orginalsigplot_1, sampsigplot_1) = plt.subplots(2, 1)
# fig.suptitle('Signal 1 (300 MHz)')
# orginalsigplot_1.plot(Sig1, Sig_1, 'r')
# orginalsigplot_1.set_xlabel('Time (s)')
# orginalsigplot_1.set_ylabel('Amplitude')
# sampsigplot_1.plot(Time_samp_1, sampsig_1, 'r')
# sampsigplot_1.set_xlabel('Time (s)')
# sampsigplot_1.set_ylabel('Amplitude')
# plt.tight_layout()
# plt.show()
#
# fig,(orginalsigplot_2, sampsigplot_2) = plt.subplots(2, 1)
# fig.suptitle('Signal 2 (800 MHz)')
# orginalsigplot_2.plot(Sig2, Sig_2, 'r')
# orginalsigplot_2.set_xlabel('Time (s)')
# orginalsigplot_2.set_ylabel('Amplitude')
# sampsigplot_2.plot(Time_samp_2, sampsig_2, 'r')
# sampsigplot_2.set_xlabel('Time (s)')
# sampsigplot_2.set_ylabel('Amplitude')
# plt.tight_layout()
# plt.show()
#
#
# #Q2.D
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# fs = 800e6
# Freq1 = 300e6
# T = 10 / Freq1
# Ts = 1 / fs
# t = np.linspace(0, 10 * T, 10000)
# X1 = np.cos(2 * np.pi * Freq1 * t)
# x_s = np.arange(0, (10 * T) - Ts, Ts)
# x_s = np.cos(2 * np.pi * Freq1 * x_s)
# t_sinc = np.linspace(0, (10 * T) - Ts, 10000)
# x_sinc = np.zeros(len(t_sinc))
# xr = np.zeros(len(t_sinc))
#
# for i in range(len(x_s)):
#     x_sinc = np.sinc((t_sinc - (i * Ts)) / Ts)
#     xr += x_s[i] * x_sinc
#
# t_s1 = np.arange(Ts / 2, (10 * T) - (Ts / 2), Ts)
# x_s1 = np.cos(2 * np.pi * Freq1 * t_s1)
# t_sinc1 = np.linspace(Ts / 2, (10 * T) - (Ts / 2), 10000)
# x_sinc1 = np.zeros(len(t_sinc1))
# xr1 = np.zeros(len(t_sinc1))
#
# for i in range(len(t_s1)):
#     x_sinc1 = np.sinc((t_sinc1 - (i * Ts)) / Ts)
#     xr1 += x_s1[i] * x_sinc1
#
# fig, (m1, m2, m3, m4, m5, m6, m7) = plt.subplots(7, 1, figsize=(10, 10))
#
# m1.plot(t, X1, color='red')
# m1.set_title('Original Signal')
# m1.set_xlabel('Time (s)')
# m1.set_ylabel('Amplitude')
#
# m2.plot(x_s, x_s, color='red')
# m2.set_title('Sampled Signal at Ts')
# m2.set_xlabel('Time (s)')
# m2.set_ylabel('Amplitude')
#
# m3.plot(t_sinc, x_sinc, color='red')
# m3.set_title('Sinc Function for Reconstruction')
# m3.set_xlabel('Time (s)')
# m3.set_ylabel('Amplitude')
#
# m4.plot(t_sinc, xr, color='red')
# m4.set_title('Reconstructed Signal from Ts Sampling')
# m4.set_xlabel('Time (s)')
# m4.set_ylabel('Amplitude')
#
# m5.plot(t_s1, x_s1, color='red')
# m5.set_title('Sampled Signal at Ts/2 Shift')
# m5.set_xlabel('Time (s)')
# m5.set_ylabel('Amplitude')
#
# m6.plot(t_sinc1, x_sinc1, color='red')
# m6.set_title('Sinc Function for Ts/2 Shift')
# m6.set_xlabel('Time (s)')
# m6.set_ylabel('Amplitude')
#
# m7.plot(t_sinc1, xr1, color='red')
# m7.set_title('Reconstructed Signal from Ts/2 Shift Sampling')
# m7.set_xlabel('Time (s)')
# m7.set_ylabel('Amplitude')
#
# def MeanSquareError(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)
#
# ms_error = MeanSquareError(xr1, X1)
# print(f'Mean Squared Error (MSE) for Ts/2 Shift Sampling: {ms_error}')
#
# plt.tight_layout()
# plt.show()
#
# #Q2. E
#
#
# # for Fs = 1000Mhz
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# fs = 1000e6
# Freq1 = 300e6
# T = 10 / Freq1
# Ts = 1 / fs
# t = np.linspace(0, 10 * T, 10000)
# X1 = np.cos(2 * np.pi * Freq1 * t)
# x_s = np.arange(0, (10 * T) - Ts, Ts)
# x_s = np.cos(2 * np.pi * Freq1 * x_s)
# t_sinc = np.linspace(0, (10 * T) - Ts, 10000)
# x_sinc = np.zeros(len(t_sinc))
# xr = np.zeros(len(t_sinc))
#
# for i in range(len(x_s)):
#     x_sinc = np.sinc((t_sinc - (i * Ts)) / Ts)
#     xr += x_s[i] * x_sinc
#
# t_s1 = np.arange(Ts / 2, (10 * T) - (Ts / 2), Ts)
# x_s1 = np.cos(2 * np.pi * Freq1 * t_s1)
# t_sinc1 = np.linspace(Ts / 2, (10 * T) - (Ts / 2), 10000)
# x_sinc1 = np.zeros(len(t_sinc1))
# xr1 = np.zeros(len(t_sinc1))
#
# for i in range(len(t_s1)):
#     x_sinc1 = np.sinc((t_sinc1 - (i * Ts)) / Ts)
#     xr1 += x_s1[i] * x_sinc1
#
# fig, (m1, m2, m3, m4, m5, m6, m7) = plt.subplots(7, 1, figsize=(10, 10))
#
# m1.plot(t, X1, color='red')
# m1.set_title('Original Signal')
# m1.set_xlabel('Time (s)')
# m1.set_ylabel('Amplitude')
#
# m2.plot(x_s, x_s, color='red')
# m2.set_title('Sampled Signal at Ts')
# m2.set_xlabel('Time (s)')
# m2.set_ylabel('Amplitude')
#
# m3.plot(t_sinc, x_sinc, color='red')
# m3.set_title('Sinc Function for Reconstruction')
# m3.set_xlabel('Time (s)')
# m3.set_ylabel('Amplitude')
#
# m4.plot(t_sinc, xr, color='red')
# m4.set_title('Reconstructed Signal from Ts Sampling')
# m4.set_xlabel('Time (s)')
# m4.set_ylabel('Amplitude')
#
# m5.plot(t_s1, x_s1, color='red')
# m5.set_title('Sampled Signal at Ts/2 Shift')
# m5.set_xlabel('Time (s)')
# m5.set_ylabel('Amplitude')
#
# m6.plot(t_sinc1, x_sinc1, color='red')
# m6.set_title('Sinc Function for Ts/2 Shift')
# m6.set_xlabel('Time (s)')
# m6.set_ylabel('Amplitude')
#
# m7.plot(t_sinc1, xr1, color='red')
# m7.set_title('Reconstructed Signal from Ts/2 Shift Sampling')
# m7.set_xlabel('Time (s)')
# m7.set_ylabel('Amplitude')
#
# def MeanSquareError(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)
#
# ms_error = MeanSquareError(xr1, X1)
# print(f'Mean Squared Error (MSE) for Ts/2 Shift Sampling: {ms_error}')
#
# plt.tight_layout()
# plt.show()





#for Fs - 500 Mhz

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# fs = 500e6
# Freq1 = 300e6
# T = 10 / Freq1
# Ts = 1 / fs
# t = np.linspace(0, 10 * T, 10000)
# X1 = np.cos(2 * np.pi * Freq1 * t)
# x_s = np.arange(0, (10 * T) - Ts, Ts)
# x_s = np.cos(2 * np.pi * Freq1 * x_s)
# t_sinc = np.linspace(0, (10 * T) - Ts, 10000)
# x_sinc = np.zeros(len(t_sinc))
# xr = np.zeros(len(t_sinc))
#
# for i in range(len(x_s)):
#     x_sinc = np.sinc((t_sinc - (i * Ts)) / Ts)
#     xr += x_s[i] * x_sinc
#
# t_s1 = np.arange(Ts / 2, (10 * T) - (Ts / 2), Ts)
# x_s1 = np.cos(2 * np.pi * Freq1 * t_s1)
# t_sinc1 = np.linspace(Ts / 2, (10 * T) - (Ts / 2), 10000)
# x_sinc1 = np.zeros(len(t_sinc1))
# xr1 = np.zeros(len(t_sinc1))
#
# for i in range(len(t_s1)):
#     x_sinc1 = np.sinc((t_sinc1 - (i * Ts)) / Ts)
#     xr1 += x_s1[i] * x_sinc1
#
# fig, (m1, m2, m3, m4, m5, m6, m7) = plt.subplots(7, 1, figsize=(10, 10))
#
# m1.plot(t, X1, color='red')
# m1.set_title('Original Signal')
# m1.set_xlabel('Time (s)')
# m1.set_ylabel('Amplitude')
#
# m2.plot(x_s, x_s, color='red')
# m2.set_title('Sampled Signal at Ts')
# m2.set_xlabel('Time (s)')
# m2.set_ylabel('Amplitude')
#
# m3.plot(t_sinc, x_sinc, color='red')
# m3.set_title('Sinc Function for Reconstruction')
# m3.set_xlabel('Time (s)')
# m3.set_ylabel('Amplitude')
#
# m4.plot(t_sinc, xr, color='red')
# m4.set_title('Reconstructed Signal from Ts Sampling')
# m4.set_xlabel('Time (s)')
# m4.set_ylabel('Amplitude')
#
# m5.plot(t_s1, x_s1, color='red')
# m5.set_title('Sampled Signal at Ts/2 Shift')
# m5.set_xlabel('Time (s)')
# m5.set_ylabel('Amplitude')
#
# m6.plot(t_sinc1, x_sinc1, color='red')
# m6.set_title('Sinc Function for Ts/2 Shift')
# m6.set_xlabel('Time (s)')
# m6.set_ylabel('Amplitude')
#
# m7.plot(t_sinc1, xr1, color='red')
# m7.set_title('Reconstructed Signal from Ts/2 Shift Sampling')
# m7.set_xlabel('Time (s)')
# m7.set_ylabel('Amplitude')
#
# def MeanSquareError(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)
#
# ms_error = MeanSquareError(xr1, X1)
# print(f'Mean Squared Error (MSE) for Ts/2 Shift Sampling: {ms_error}')
#
# plt.tight_layout()
# plt.show()


#Q3

#A)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
#
# sig_freq = 2e6
# f_samp = 5e6
# samp_period = 1 / f_samp
# time = np.linspace(0, 10 * (1 / sig_freq), 10000)
# time_samp = np.arange(0, 10 * (1 / sig_freq), samp_period)
# org_sig = np.cos(2 * np.pi * sig_freq * time)
# Samp_Signal = np.cos(2 * np.pi * sig_freq * time_samp)
# Sig_dft = fft(Samp_Signal, 64)
# Mag_dft = np.abs(Sig_dft)
#
# fig, (m1, m2, m3) = plt.subplots(3, 1, figsize=(10, 8))
#
# m1.plot(time, org_sig, color='red')
# m1.set_title('Original Signal: $x(t) = \cos(2 \pi f t)$')
# m1.set_xlabel('Time [s]')
# m1.set_ylabel('Amplitude')
#
# m2.plot(time_samp, Samp_Signal, color='blue')
# m2.set_title('Sampled Signal: $x_s[n]$')
# m2.set_xlabel('Time [s]')
# m2.set_ylabel('Amplitude')
#
# frequency_axis = np.fft.fftfreq(64, samp_period)
# m3.plot(frequency_axis, Mag_dft, color='green')
# m3.set_title('Magnitude of the 64-point DFT')
# m3.set_xlabel('Frequency [Hz]')
# m3.set_ylabel('Magnitude')
#
# plt.tight_layout()
# plt.show()



#Q3.B
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
#
# freq1 = 200e6
# freq2 = 400e6
# samp_freq = 1e9
# org_time = np.linspace(0, 10 * (1 / (freq1 + freq2)), 10000)
# samp_time = np.arange(0, 10 * (1 / (freq1 + freq2)), 1 / samp_freq)
# org_sig = np.cos(2 * np.pi * freq1 * org_time) + np.cos(2 * np.pi * freq2 * org_time)
# samp_sig = np.cos(2 * np.pi * freq1 * samp_time) + np.cos(2 * np.pi * freq2 * samp_time)
# sig_dft = fft(samp_sig, 64)
# mag_dft = np.abs(sig_dft)
# freqz = np.fft.fftfreq(64, 1/samp_freq)
# pos_freqz = freqz[:32]
# pos_mag_dft = mag_dft[:32]
#
# fig, (m1, m2, m3) = plt.subplots(3, 1, figsize=(10, 8))
#
# m1.plot(org_time, org_sig, color='red')
# m1.set_title('Original Signal: $y(t) = \cos(2\pi F_1 t) + \cos(2\pi F_2 t)$')
# m1.set_xlabel('Time [s]')
# m1.set_ylabel('Amplitude')
#
# m2.stem(samp_time, samp_sig, use_line_collection=True, linefmt='r', markerfmt='ro')
# m2.set_title('Sampled Signal at $F_s = 1$ GHz')
# m2.set_xlabel('Time [s]')
# m2.set_ylabel('Amplitude')
#
# m3.plot(pos_freqz, pos_mag_dft, color='red')
# m3.set_title('Magnitude of 64-point DFT (Positive Frequencies)')
# m3.set_xlabel('Frequency [Hz]')
# m3.set_ylabel('Magnitude')
#
# plt.tight_layout()
# plt.show()


#Q3. C

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
#
# freq1 = 200e6
# freq2 = 400e6
# f_samp = 500e6
# org_time = np.linspace(0, 10 * (1 / (freq1 + freq2)), 10000)
# samp_time = np.arange(0, 10 * (1 / (freq1 + freq2)), 1 / f_samp)
# org_sig = np.cos(2 * np.pi * freq1 * org_time) + np.cos(2 * np.pi * freq2 * org_time)
# sig_sampled = np.cos(2 * np.pi * freq1 * samp_time) + np.cos(2 * np.pi * freq2 * samp_time)
# sig_dft = fft(sig_sampled, 64)
# mag_dft = np.abs(sig_dft)
# freqz = np.fft.fftfreq(64, 1/f_samp)
# pos_freqz = freqz[:32]
# pos_mag_dft = mag_dft[:32]
#
# fig, (m1, ax1, m3) = plt.subplots(3, 1, figsize=(10, 8))
#
# m1.plot(org_time, org_sig, color='red')
# m1.set_title('Original Signal: $y(t) = \cos(2\pi F_1 t) + \cos(2\pi F_2 t)$')
# m1.set_xlabel('Time [s]')
# m1.set_ylabel('Amplitude')
#
# ax1.stem(samp_time, sig_sampled, use_line_collection=True, linefmt='r', markerfmt='ro')
# ax1.set_title('Sampled Signal at $F_s = 500$ MHz')
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('Amplitude')
#
# m3.plot(pos_freqz, pos_mag_dft, color='red')
# m3.set_title('Magnitude of 64-point DFT (Positive Frequencies)')
# m3.set_xlabel('Frequency [Hz]')
# m3.set_ylabel('Magnitude')
#
# plt.tight_layout()
# plt.show()



#Q4.

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft, fftshift
#
# freq1 = 2e6
# samp_freq = 5e6
# org_time = np.linspace(0, 10 * (1 / freq1), 10000)
# samp_time = np.arange(0, 10 * (1 / freq1), 1 / samp_freq)
# signal_org = np.cos(2 * np.pi * freq1 * org_time)
# sig_sampled = np.cos(2 * np.pi * freq1 * samp_time)
# sig_dft = fft(sig_sampled, 50)
# mag_dft = np.abs(sig_dft)
# black_win = np.blackman(len(mag_dft))
# dft_wind = mag_dft * black_win
# freqz = np.fft.fftfreq(len(mag_dft), 1/samp_freq)
# freqz = fftshift(freqz)
# resp_in_db = 20 * np.log10(np.abs(fftshift(dft_wind / abs(dft_wind).max())))
#
# fig, (m1, m2, m3, m4) = plt.subplots(4, 1, figsize=(10, 8))
#
# m1.plot(org_time, signal_org, color='red')
# m1.set_title('Original Signal: $x(t) = \cos(2\pi f_1 t)$')
# m1.set_xlabel('Time in s')
# m1.set_ylabel('Amplitude')
#
# m2.plot(samp_time, sig_sampled, color='red')
# m2.set_title('Sampled Signal at $F_s = 5$ MHz')
# m2.set_xlabel('Time in s')
# m2.set_ylabel('Amplitude')
#
# m3.plot(freqz, resp_in_db, color='red')
# m3.set_title('Blackman Windowed Frequency Response')
# m3.set_xlabel('Frequency in Hz')
# m3.set_ylabel('Magnitude in dB')
#
# m4.plot(freqz, mag_dft, color='red')
# m4.set_title('Magnitude of the DFT of Sampled Signal')
# m4.set_xlabel('Frequency in Hz')
# m4.set_ylabel('Magnitude')
#
# plt.tight_layout()
# plt.show()

