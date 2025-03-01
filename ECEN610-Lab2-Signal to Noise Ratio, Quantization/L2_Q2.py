#Q2

#Q2.A

#For Period= 30

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


Freq = 200e6
period = 1/Freq
timevect = np.linspace(0, 10*period, 1000)
org_sig = np.cos(2*np.pi*Freq * timevect)


samp_freq = 400e6
samp_pd = 1/samp_freq
samp_timevect = np.arange(0, (30*period), samp_pd)
samp_signal = np.cos(2*np.pi*Freq * samp_timevect)


def quant(signal, bits):
    levels = 2**bits
    steps = (1) / levels
    quant_sig = (np.round(signal/steps) * steps)
    return quant_sig


quant_op = quant(samp_signal, 6)


def psd_plot(signal_data, samp_freq):
    Freq, den = signal.periodogram(signal_data, samp_freq, nfft=None)

    psd_max_val = np.max(den)
    den_scaled = den * (200 / psd_max_val)
    plt.plot(Freq, den_scaled, color='green', label="PSD (Quantized Signal)")
    plt.xlabel("frequency in Hz")
    plt.ylabel("PSD of Signal (Scaled V^2/Hz)")
    plt.title("PSD of Quantized Signal")
    plt.legend(loc='upper right')


plt.figure(figsize=(10, 6))
plt.plot(timevect, org_sig, color='green', label="Sine Waveform")
plt.xlabel("Time in Seconds")
plt.ylabel("Original Signal Amplitude")
plt.title('The Original Signal Plot')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.stem(samp_timevect, samp_signal, basefmt=" ", linefmt='g', markerfmt='go', use_line_collection=True, label="Sampled Signal")
plt.xlabel("Time in Seconds")
plt.ylabel("Sampled Signal Amplitude")
plt.title('The Sampled Signal Plot')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(samp_timevect, quant_op, color='green', label="Quantized Signal")
plt.xlabel("Time in Seconds")
plt.ylabel("Quantized Signal Amplitude")
plt.title('The Quantized Signal')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
psd_plot(quant_op, samp_freq)
plt.show()


levels = 2**6
Noise_Sig = quant_op - np.round(quant_op*(levels-1)/2)/(levels-1)*2


Freq, den = signal.periodogram(quant_op, samp_freq, nfft=None)
signal_power = den[np.argmax(den)]

Freq, den_noise = signal.periodogram(Noise_Sig, samp_freq, nfft=None)
noise_power = den_noise[np.argmax(den_noise)]

print("Obtained Signal Power:", signal_power)
print("Obtained Noise Power:", noise_power)

SNR_actual = 10*np.log10(signal_power/noise_power)
print(f'SNR Computed = {SNR_actual} dB')



#For Period = 100

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


Freq = 200e6
period = 1/Freq
timevect = np.linspace(0, 10*period, 1000)
org_sig = np.cos(2*np.pi*Freq * timevect)


samp_freq = 400e6
samp_pd = 1/samp_freq
samp_timevect = np.arange(0, (100*period), samp_pd)
samp_signal = np.cos(2*np.pi*Freq * samp_timevect)


def quant(signal, bits):
    levels = 2**bits
    steps = (1) / levels
    quant_sig = (np.round(signal/steps) * steps)
    return quant_sig


quant_op = quant(samp_signal, 6)


def psd_plot(signal_data, samp_freq):
    Freq, den = signal.periodogram(signal_data, samp_freq, nfft=None)

    psd_max_val = np.max(den)
    den_scaled = den * (200 / psd_max_val)
    plt.plot(Freq, den_scaled, color='green', label="PSD (Quantized Signal)")
    plt.xlabel("frequency in Hz")
    plt.ylabel("PSD of Signal (Scaled V^2/Hz)")
    plt.title("PSD of Quantized Signal")
    plt.legend(loc='upper right')


plt.figure(figsize=(10, 6))
plt.plot(timevect, org_sig, color='green', label="Sine Waveform")
plt.xlabel("Time in Seconds")
plt.ylabel("Original Signal Amplitude")
plt.title('The Original Signal Plot')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.stem(samp_timevect, samp_signal, basefmt=" ", linefmt='g', markerfmt='go', use_line_collection=True, label="Sampled Signal")
plt.xlabel("Time in Seconds")
plt.ylabel("Sampled Signal Amplitude")
plt.title('The Sampled Signal Plot')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(samp_timevect, quant_op, color='green', label="Quantized Signal")
plt.xlabel("Time in Seconds")
plt.ylabel("Quantized Signal Amplitude")
plt.title('The Quantized Signal')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
psd_plot(quant_op, samp_freq)
plt.show()

levels = 2**6
Noise_Sig = quant_op - np.round(quant_op*(levels-1)/2)/(levels-1)*2


Freq, den = signal.periodogram(quant_op, samp_freq, nfft=None)
signal_power = den[np.argmax(den)]

Freq, den_noise = signal.periodogram(Noise_Sig, samp_freq, nfft=None)
noise_power = den_noise[np.argmax(den_noise)]

print("Obtained Signal Power:", signal_power)
print("Obtained Noise Power:", noise_power)

SNR_actual = 10*np.log10(signal_power/noise_power)
print(f'SNR Computed = {SNR_actual} dB')








#Q2.B

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


Freq = 200e6
period = 1/Freq
timevect = np.linspace(0, 10*period, 1000)
org_sig = np.cos(2*np.pi*Freq * timevect)


samp_freq = 450e6
samp_pd = 1/samp_freq
samp_timevect = np.arange(0, (30*period), samp_pd)
samp_signal = np.cos(2*np.pi*Freq * samp_timevect)


def quant(signal, bits):
    levels = 2**bits
    steps = (1) / levels
    quant_sig = (np.round(signal/steps) * steps)
    return quant_sig


quant_op = quant(samp_signal, 6)


def psd_plot(signal_data, samp_freq):
    Freq, den = signal.periodogram(signal_data, samp_freq, nfft=None)

    psd_max_val = np.max(den)
    den_scaled = den * (200 / psd_max_val)
    plt.plot(Freq, den_scaled, color='green', label="PSD (Quantized Signal)")
    plt.xlabel("frequency in Hz")
    plt.ylabel("PSD of Signal (Scaled V^2/Hz)")
    plt.title("PSD of Quantized Signal")
    plt.legend(loc='upper right')


plt.figure(figsize=(10, 6))
plt.plot(timevect, org_sig, color='green', label="Sine Waveform")
plt.xlabel("Time in Seconds")
plt.ylabel("Original Signal Amplitude")
plt.title('The Original Signal Plot')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.stem(samp_timevect, samp_signal, basefmt=" ", linefmt='g', markerfmt='go', use_line_collection=True, label="Sampled Signal")
plt.xlabel("Time in Seconds")
plt.ylabel("Sampled Signal Amplitude")
plt.title('The Sampled Signal Plot')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(samp_timevect, quant_op, color='green', label="Quantized Signal")
plt.xlabel("Time in Seconds")
plt.ylabel("Quantized Signal Amplitude")
plt.title('The Quantized Signal')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
psd_plot(quant_op, samp_freq)
plt.show()


levels = 2**6
Noise_Sig = quant_op - np.round(quant_op*(levels-1)/2)/(levels-1)*2


Freq, den = signal.periodogram(quant_op, samp_freq, nfft=None)
signal_power = den[np.argmax(den)]

Freq, den_noise = signal.periodogram(Noise_Sig, samp_freq, nfft=None)
noise_power = den_noise[np.argmax(den_noise)]

print("Obtained Signal Power:", signal_power)
print("Obtained Noise Power:", noise_power)

SNR_actual = 10*np.log10(signal_power/noise_power)
print(f'SNR Computed = {SNR_actual} dB')




#Q2.C

#Repeat A, but with N= 12, Periods= 30, 100

#for N=12 and period = 30



import matplotlib.pyplot as plt
import numpy as np  
from scipy import signal

b_count = 12
N = 30


frequency = 200e6
pd = 1 / frequency
t = np.linspace(0, 10 * pd, 1000)
org_sig = np.cos(2 * np.pi * frequency * t)


plt.figure()
plt.plot(t, org_sig, color='green', label="Original Signal")
plt.ylabel("Original signal")
plt.title('Original signal')
plt.legend(loc='upper right')  
plt.show()


samp_freq = 400e6
samp_pd = 1 / samp_freq
ts = np.arange(0, (N * pd), samp_pd)
samp_sig = np.cos(2 * np.pi * frequency * ts)


plt.figure()
plt.stem(ts, samp_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Sampled Signal")
plt.ylabel("Sampled signal")
plt.title('Sampled signal')
plt.legend(loc='upper right')  
plt.show()


def quant(signal, b_count):
    levels = 2 ** b_count
    q_steps = 1 / levels
    quant_sig = np.round(signal / q_steps) * q_steps
    return quant_sig

quant_sig = quant(samp_sig, b_count)


plt.figure()
plt.stem(ts, quant_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Quantized Signal")
plt.ylabel("Quantized signal")
plt.title('Quantized signal')
plt.legend(loc='upper right')  
plt.show()


levels = 2 ** b_count
noise_sig = quant_sig - np.round(quant_sig * (levels - 1) / 2) / (levels - 1) * 2


f, den = signal.periodogram(quant_sig, samp_freq)


n = len(ts)
fft_quant = np.fft.fft(quant_sig, n)  
psd = fft_quant * np.conj(fft_quant) / n
freq_signal = (samp_freq / n) * np.arange(n)  

plt.figure()
plt.plot(freq_signal, np.abs(psd), color='green', label="PSD of Quantized Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("PSD (Quantized Signal)")
plt.legend(loc='upper right')  
plt.show()


f_max = f[np.argmax(den)]
sig_pwr = 0
noise_pwr = 0
for i in range(len(f)):
    if f[i] == f_max:
        sig_pwr += den[i]

f, den = signal.periodogram(noise_sig, samp_freq)
f_max = f[np.argmax(den)]
for i in range(len(f)):
    if f[i] == f_max:
        noise_pwr += den[i]

print("Signal power", sig_pwr)
print("Noise power", noise_pwr)


actualSNR = 10 * np.log10(sig_pwr / noise_pwr)
print('SNR obtained =', actualSNR, 'dB')



# When N=12 and Period = 100




import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

b_count = 12
N = 100


frequency = 200e6
pd = 1 / frequency
t = np.linspace(0, 10 * pd, 1000)
org_sig = np.cos(2 * np.pi * frequency * t)


plt.figure()
plt.plot(t, org_sig, color='green', label="Cosine Waveform")
plt.ylabel("Original signal Amplitude")
plt.title('The Original signal Plot')
plt.legend(loc='upper right')
plt.show()


samp_freq = 400e6
samp_pd = 1 / samp_freq
ts = np.arange(0, (N * pd), samp_pd)
samp_sig = np.cos(2 * np.pi * frequency * ts)


plt.figure()
plt.stem(ts, samp_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Sampled Signal")
plt.ylabel("Sampled signal Amplitude")
plt.title('The Sampled signal Signal')
plt.legend(loc='upper right')
plt.show()


def quant(signal, b_count):
    levels = 2 ** b_count
    q_steps = 1 / levels
    quant_sig = np.round(signal / q_steps) * q_steps
    return quant_sig

quant_sig = quant(samp_sig, b_count)


plt.figure()
plt.stem(ts, quant_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Quantized Signal")
plt.ylabel("Quantized signal Amplitude")
plt.title('The Quantized signal')
plt.legend(loc='upper right')
plt.show()


levels = 2 ** b_count
noise_sig = quant_sig - np.round(quant_sig * (levels - 1) / 2) / (levels - 1) * 2


f, den = signal.periodogram(quant_sig, samp_freq)


n = len(ts)
fft_quant = np.fft.fft(quant_sig, n)
psd = fft_quant * np.conj(fft_quant) / n
freq_signal = (samp_freq / n) * np.arange(n)

plt.figure()
plt.plot(freq_signal, np.abs(psd), color='green', label="PSD (Quantized Signal)")
plt.xlabel("frequency in Hz")
plt.ylabel("PSD")
plt.title("PSD of the Quantized Signal")
plt.legend(loc='upper right')
plt.show()


f_max = f[np.argmax(den)]
sig_pwr = 0
noise_pwr = 0
for i in range(len(f)):
    if f[i] == f_max:
        sig_pwr += den[i]

f, den = signal.periodogram(noise_sig, samp_freq)
f_max = f[np.argmax(den)]
for i in range(len(f)):
    if f[i] == f_max:
        noise_pwr += den[i]

print("Obtained Signal power", sig_pwr)
print("Obtained Noise power", noise_pwr)


actualSNR = 10 * np.log10(sig_pwr / noise_pwr)
print('SNR Computed =', actualSNR, 'dB')



#Q2.D
#When N = 6



import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
b_count = 6
Frequency = 200e6
pd = 1 / Frequency
t = np.linspace(0, 10 * pd, 1000)
org_sig = np.cos(2 * np.pi * Frequency * t)
plt.figure()
plt.plot(t, org_sig, color='green', label="Cosine Waveform")
plt.ylabel("Original signal Amplitude")
plt.title('The Original signal Plot')
plt.legend(loc='upper right')
plt.show()
samp_freq = 400e6
samp_pd = 1 / samp_freq
ts = np.arange(0, (100 * pd), samp_pd)
samp_sig = np.cos(2 * np.pi * Frequency * ts)
plt.figure()
plt.stem(ts, samp_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Sampled Waveform")
plt.ylabel("Sampled signal Amplitude")
plt.title('The Sampled signal Plot')
plt.legend(loc='upper right')
plt.show()
def hann_wind(signal):
    N = len(signal)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) / (N - 1)))
    return signal * window
quant_sig = hann_wind(samp_sig)
plt.figure()
plt.stem(ts, quant_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Hanning Window Sampled Signal")
plt.ylabel("Sampled signal Amplitude")
plt.title('Sampled signal of Hanning window')
plt.legend(loc='upper right')
plt.show()
def quant(signal, b_count):
    q_levels = 2 ** b_count
    q_step = 1 / q_levels
    q_sig = np.round(signal / q_step) * q_step
    return q_sig
Hann_win_q_sig = quant(quant_sig, b_count)
plt.figure()
plt.stem(ts, Hann_win_q_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Hanning Window Quantized Signal")
plt.ylabel("Quantized Signal Amplitude")
plt.title('Quantized Signal of Hanning Window')
plt.legend(loc='upper right')
plt.show()
q_levels = 2 ** b_count
noise_sig = Hann_win_q_sig - np.round(Hann_win_q_sig * (q_levels - 1) / 2) / (q_levels - 1) * 2
n = len(ts)
fft_q = np.fft.fft(Hann_win_q_sig, n)
psd_q = fft_q * np.conj(fft_q) / n
freq_signal = (samp_freq / n) * np.arange(n)
plt.figure()
plt.plot(freq_signal, np.abs(psd_q), color='green', label="PSD of Quantized Signal")
plt.xlabel("frequency in Hz")
plt.ylabel("PSD")
plt.title("PSD of Quantized Signal")
plt.legend(loc='upper right')
plt.show()
f, den = signal.periodogram(Hann_win_q_sig, samp_freq)
f_maxx = f[np.argmax(den)]
sig_pwr = 0
noise_pwr = 0
for i in range(len(f)):
    if f[i] == f_maxx:
        sig_pwr += den[i]
f, den = signal.periodogram(noise_sig, samp_freq)
f_maxx = f[np.argmax(den)]
for i in range(len(f)):
    if f[i] == f_maxx:
        noise_pwr += den[i]
print("Obtained Signal power", sig_pwr)
print("Obtained Noise power", noise_pwr)
SNR_Actual = 10 * np.log10(np.abs(sig_pwr) / np.abs(noise_pwr))
print('SNR Computed =', str(np.abs(SNR_Actual)), 'dB')



#When N = 12


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
b_count = 12
Frequency = 200e6
pd = 1 / Frequency
t = np.linspace(0, 10 * pd, 1000)
org_sig = np.cos(2 * np.pi * Frequency * t)

plt.figure()
plt.plot(t, org_sig, color='green', label="Cosine Waveform")
plt.ylabel("Original signal Amplitude")
plt.title('The Original signal Plot')
plt.legend(loc='upper right')
plt.show()

samp_freq = 400e6
samp_pd = 1 / samp_freq
ts = np.arange(0, (100 * pd), samp_pd)
samp_sig = np.cos(2 * np.pi * Frequency * ts)

plt.figure()
plt.stem(ts, samp_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Sampled Waveform")
plt.ylabel("Sampled signal Amplitude")
plt.title('The Sampled signal Plot')
plt.legend(loc='upper right')
plt.show()

def hann_wind(signal):
    N = len(signal)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) / (N - 1)))
    return signal * window

quant_sig = hann_wind(samp_sig)

plt.figure()
plt.stem(ts, quant_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Hanning Window Sampled Signal")
plt.ylabel("Sampled signal Amplitude")
plt.title('Sampled signal of Hanning window')
plt.legend(loc='upper right')
plt.show()

def quant(signal, b_count):
    q_levels = 2 ** b_count
    q_step = 1 / q_levels
    q_sig = np.round(signal / q_step) * q_step
    return q_sig

Hann_win_q_sig = quant(quant_sig, b_count)

plt.figure()
plt.stem(ts, Hann_win_q_sig, basefmt=" ", linefmt='green', markerfmt='go', label="Hanning Window Quantized Signal")
plt.ylabel("Quantized Signal Amplitude")
plt.title('Quantized Signal of Hanning Window')
plt.legend(loc='upper right')
plt.show()

q_levels = 2 ** b_count
noise_sig = Hann_win_q_sig - np.round(Hann_win_q_sig * (q_levels - 1) / 2) / (q_levels - 1) * 2

n = len(ts)
fft_q = np.fft.fft(Hann_win_q_sig, n)
psd_q = fft_q * np.conj(fft_q) / n
freq_signal = (samp_freq / n) * np.arange(n)

plt.figure()
plt.plot(freq_signal, np.abs(psd_q), color='green', label="PSD of Quantized Signal")
plt.xlabel("frequency in Hz")
plt.ylabel("PSD")
plt.title("PSD of Quantized Signal")
plt.legend(loc='upper right')
plt.show()

f, den = signal.periodogram(Hann_win_q_sig, samp_freq)
f_maxx = f[np.argmax(den)]
sig_pwr = 0
noise_pwr = 0

for i in range(len(f)):
    if f[i] == f_maxx:
        sig_pwr += den[i]
f, den = signal.periodogram(noise_sig, samp_freq)
f_maxx = f[np.argmax(den)]
for i in range(len(f)):
    if f[i] == f_maxx:
        noise_pwr += den[i]
print("Obtained Signal power", sig_pwr)
print("Obtained Noise power", noise_pwr)
SNR_Actual = 10 * np.log10(np.abs(sig_pwr) / np.abs(noise_pwr))
print('SNR Computed =', str(np.abs(SNR_Actual)), 'dB')
