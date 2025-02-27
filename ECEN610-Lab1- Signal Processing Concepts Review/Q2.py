#Q2



#Q2.A


import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 500e6
f_1 = 300e6
f_2 = 800e6

Sig1 = np.linspace(0, 10 / f_1, 15000)
Sig2 = np.linspace(0, 10 / f_2, 15000)

Time_samp_1 = np.arange(0, 10 / f_1, 1 / sampling_rate)
Time_samp_2 = np.arange(0, 10 / f_2, 1 / sampling_rate)

Sig_1 = np.cos(2 * np.pi * f_1 * Sig1)
Sig_2 = np.cos(2 * np.pi * f_2 * Sig2)

sampsig_1 = np.cos(2 * np.pi * f_1 * Time_samp_1)
sampsig_2 = np.cos(2 * np.pi * f_2 * Time_samp_2)

fig, (orginalsigplot_1, sampsigplot_1) = plt.subplots(2, 1)
fig.suptitle('Signal 1 (300 MHz)')
orginalsigplot_1.plot(Sig1, Sig_1, 'r')
orginalsigplot_1.set_xlabel('Time (s)')
orginalsigplot_1.set_ylabel('Amplitude')
sampsigplot_1.plot(Time_samp_1, sampsig_1, 'r')
sampsigplot_1.set_xlabel('Time (s)')
sampsigplot_1.set_ylabel('Amplitude')
plt.tight_layout()
plt.show()

fig,(orginalsigplot_2, sampsigplot_2) = plt.subplots(2, 1)
fig.suptitle('Signal 2 (800 MHz)')
orginalsigplot_2.plot(Sig2, Sig_2, 'r')
orginalsigplot_2.set_xlabel('Time (s)')
orginalsigplot_2.set_ylabel('Amplitude')
sampsigplot_2.plot(Time_samp_2, sampsig_2, 'r')
sampsigplot_2.set_xlabel('Time (s)')
sampsigplot_2.set_ylabel('Amplitude')
plt.tight_layout()
plt.show()







#Q2.D

import numpy as np
import matplotlib.pyplot as plt

fs = 800e6
Freq1 = 300e6
T = 10 / Freq1
Ts = 1 / fs
t = np.linspace(0, 10 * T, 10000)
X1 = np.cos(2 * np.pi * Freq1 * t)
x_s = np.arange(0, (10 * T) - Ts, Ts)
x_s = np.cos(2 * np.pi * Freq1 * x_s)
t_sinc = np.linspace(0, (10 * T) - Ts, 10000)
x_sinc = np.zeros(len(t_sinc))
xr = np.zeros(len(t_sinc))

for i in range(len(x_s)):
    x_sinc = np.sinc((t_sinc - (i * Ts)) / Ts)
    xr += x_s[i] * x_sinc

t_s1 = np.arange(Ts / 2, (10 * T) - (Ts / 2), Ts)
x_s1 = np.cos(2 * np.pi * Freq1 * t_s1)
t_sinc1 = np.linspace(Ts / 2, (10 * T) - (Ts / 2), 10000)
x_sinc1 = np.zeros(len(t_sinc1))
xr1 = np.zeros(len(t_sinc1))

for i in range(len(t_s1)):
    x_sinc1 = np.sinc((t_sinc1 - (i * Ts)) / Ts)
    xr1 += x_s1[i] * x_sinc1

fig, (m1, m2, m3, m4, m5, m6, m7) = plt.subplots(7, 1, figsize=(10, 10))

m1.plot(t, X1, color='red')
m1.set_title('Original Signal')
m1.set_xlabel('Time (s)')
m1.set_ylabel('Amplitude')

m2.plot(x_s, x_s, color='red')
m2.set_title('Sampled Signal at Ts')
m2.set_xlabel('Time (s)')
m2.set_ylabel('Amplitude')

m3.plot(t_sinc, x_sinc, color='red')
m3.set_title('Sinc Function for Reconstruction')
m3.set_xlabel('Time (s)')
m3.set_ylabel('Amplitude')

m4.plot(t_sinc, xr, color='red')
m4.set_title('Reconstructed Signal from Ts Sampling')
m4.set_xlabel('Time (s)')
m4.set_ylabel('Amplitude')

m5.plot(t_s1, x_s1, color='red')
m5.set_title('Sampled Signal at Ts/2 Shift')
m5.set_xlabel('Time (s)')
m5.set_ylabel('Amplitude')

m6.plot(t_sinc1, x_sinc1, color='red')
m6.set_title('Sinc Function for Ts/2 Shift')
m6.set_xlabel('Time (s)')
m6.set_ylabel('Amplitude')

m7.plot(t_sinc1, xr1, color='red')
m7.set_title('Reconstructed Signal from Ts/2 Shift Sampling')
m7.set_xlabel('Time (s)')
m7.set_ylabel('Amplitude')

def MeanSquareError(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

ms_error = MeanSquareError(xr1, X1)
print(f'Mean Squared Error (MSE) for Ts/2 Shift Sampling: {ms_error}')

plt.tight_layout()
plt.show()






#Q2. E


# for Fs = 1000Mhz


import numpy as np
import matplotlib.pyplot as plt

fs = 1000e6
Freq1 = 300e6
T = 10 / Freq1
Ts = 1 / fs
t = np.linspace(0, 10 * T, 10000)
X1 = np.cos(2 * np.pi * Freq1 * t)
x_s = np.arange(0, (10 * T) - Ts, Ts)
x_s = np.cos(2 * np.pi * Freq1 * x_s)
t_sinc = np.linspace(0, (10 * T) - Ts, 10000)
x_sinc = np.zeros(len(t_sinc))
xr = np.zeros(len(t_sinc))

for i in range(len(x_s)):
    x_sinc = np.sinc((t_sinc - (i * Ts)) / Ts)
    xr += x_s[i] * x_sinc

t_s1 = np.arange(Ts / 2, (10 * T) - (Ts / 2), Ts)
x_s1 = np.cos(2 * np.pi * Freq1 * t_s1)
t_sinc1 = np.linspace(Ts / 2, (10 * T) - (Ts / 2), 10000)
x_sinc1 = np.zeros(len(t_sinc1))
xr1 = np.zeros(len(t_sinc1))

for i in range(len(t_s1)):
    x_sinc1 = np.sinc((t_sinc1 - (i * Ts)) / Ts)
    xr1 += x_s1[i] * x_sinc1

fig, (m1, m2, m3, m4, m5, m6, m7) = plt.subplots(7, 1, figsize=(10, 10))

m1.plot(t, X1, color='red')
m1.set_title('Original Signal')
m1.set_xlabel('Time (s)')
m1.set_ylabel('Amplitude')

m2.plot(x_s, x_s, color='red')
m2.set_title('Sampled Signal at Ts')
m2.set_xlabel('Time (s)')
m2.set_ylabel('Amplitude')

m3.plot(t_sinc, x_sinc, color='red')
m3.set_title('Sinc Function for Reconstruction')
m3.set_xlabel('Time (s)')
m3.set_ylabel('Amplitude')

m4.plot(t_sinc, xr, color='red')
m4.set_title('Reconstructed Signal from Ts Sampling')
m4.set_xlabel('Time (s)')
m4.set_ylabel('Amplitude')

m5.plot(t_s1, x_s1, color='red')
m5.set_title('Sampled Signal at Ts/2 Shift')
m5.set_xlabel('Time (s)')
m5.set_ylabel('Amplitude')

m6.plot(t_sinc1, x_sinc1, color='red')
m6.set_title('Sinc Function for Ts/2 Shift')
m6.set_xlabel('Time (s)')
m6.set_ylabel('Amplitude')

m7.plot(t_sinc1, xr1, color='red')
m7.set_title('Reconstructed Signal from Ts/2 Shift Sampling')
m7.set_xlabel('Time (s)')
m7.set_ylabel('Amplitude')

def MeanSquareError(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

ms_error = MeanSquareError(xr1, X1)
print(f'Mean Squared Error (MSE) for Ts/2 Shift Sampling: {ms_error}')

plt.tight_layout()
plt.show()





#for Fs - 500 Mhz


import numpy as np
import matplotlib.pyplot as plt

fs = 500e6
Freq1 = 300e6
T = 10 / Freq1
Ts = 1 / fs
t = np.linspace(0, 10 * T, 10000)
X1 = np.cos(2 * np.pi * Freq1 * t)
x_s = np.arange(0, (10 * T) - Ts, Ts)
x_s = np.cos(2 * np.pi * Freq1 * x_s)
t_sinc = np.linspace(0, (10 * T) - Ts, 10000)
x_sinc = np.zeros(len(t_sinc))
xr = np.zeros(len(t_sinc))

for i in range(len(x_s)):
    x_sinc = np.sinc((t_sinc - (i * Ts)) / Ts)
    xr += x_s[i] * x_sinc

t_s1 = np.arange(Ts / 2, (10 * T) - (Ts / 2), Ts)
x_s1 = np.cos(2 * np.pi * Freq1 * t_s1)
t_sinc1 = np.linspace(Ts / 2, (10 * T) - (Ts / 2), 10000)
x_sinc1 = np.zeros(len(t_sinc1))
xr1 = np.zeros(len(t_sinc1))

for i in range(len(t_s1)):
    x_sinc1 = np.sinc((t_sinc1 - (i * Ts)) / Ts)
    xr1 += x_s1[i] * x_sinc1

fig, (m1, m2, m3, m4, m5, m6, m7) = plt.subplots(7, 1, figsize=(10, 10))

m1.plot(t, X1, color='red')
m1.set_title('Original Signal')
m1.set_xlabel('Time (s)')
m1.set_ylabel('Amplitude')

m2.plot(x_s, x_s, color='red')
m2.set_title('Sampled Signal at Ts')
m2.set_xlabel('Time (s)')
m2.set_ylabel('Amplitude')

m3.plot(t_sinc, x_sinc, color='red')
m3.set_title('Sinc Function for Reconstruction')
m3.set_xlabel('Time (s)')
m3.set_ylabel('Amplitude')

m4.plot(t_sinc, xr, color='red')
m4.set_title('Reconstructed Signal from Ts Sampling')
m4.set_xlabel('Time (s)')
m4.set_ylabel('Amplitude')

m5.plot(t_s1, x_s1, color='red')
m5.set_title('Sampled Signal at Ts/2 Shift')
m5.set_xlabel('Time (s)')
m5.set_ylabel('Amplitude')

m6.plot(t_sinc1, x_sinc1, color='red')
m6.set_title('Sinc Function for Ts/2 Shift')
m6.set_xlabel('Time (s)')
m6.set_ylabel('Amplitude')

m7.plot(t_sinc1, xr1, color='red')
m7.set_title('Reconstructed Signal from Ts/2 Shift Sampling')
m7.set_xlabel('Time (s)')
m7.set_ylabel('Amplitude')

def MeanSquareError(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

ms_error = MeanSquareError(xr1, X1)
print(f'Mean Squared Error (MSE) for Ts/2 Shift Sampling: {ms_error}')

plt.tight_layout()
plt.show()
