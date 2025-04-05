import matplotlib.pyplot as plt
import numpy as np

freq1 = 1e9
input = 1 / freq1
time = np.linspace(0, input, 100000)  
sig = np.sin(2 * np.pi * freq1 * time)  

fig, (m1, m2, m3, m4, m5) = plt.subplots(5, 1, figsize=(10, 12))

m1.plot(time, sig, color='green')
m1.set_title('Message Signal Plot')
m1.set_xlabel('Time in S')
m1.set_ylabel('Amplitude')

F_samp = 10e9
samp_time = 1 / F_samp
constant = input / samp_time
constant2 = 100000 / constant

idealsample = [0 for d in range(100000)]
for k in range(0, 100000):
    if int(k % constant2) == 0:
        idealsample[k] = sig[k]

m2.stem(time, idealsample, linefmt='g-', markerfmt='go')  
m2.set_title('Sample Signal [No Mismatch]')
m2.set_xlabel('Time in S')
m2.set_ylabel('Amplitude')

of_err = 0.1
gain_mis = 0.707
t_mis = 2e-12
ph_off = 10

mismatc_in_sig = np.sin(2 * np.pi * freq1 * samp_time) + of_err
const3 = input / 100000
const4 = t_mis / const3

count = 0
practicalsample = [0 for d in range(100000)]
for k in range(0, 100000):
    if int(k % ((constant2 / 2) + const4)) == 0:
        count = count + 1
    if count % 2 == 0:
        practicalsample[k] = (sig[k]) * gain_mis

m3.stem(time, practicalsample, linefmt='g-', markerfmt='go')  
m3.set_title('Sample Signal with Gain Mismatch, Offset Error, Time Mismatch Plot')
m3.set_xlabel('Time in S')
m3.set_ylabel('Amplitude')

resultsignal = [0 for d in range(100000)]
for k in range(0, 100000):
    resultsignal[k] = idealsample[k] + practicalsample[k]

m4.stem(time, resultsignal, linefmt='g-', markerfmt='go')  
m4.set_title('Result Signal Plot')
m4.set_xlabel('Time in S')
m4.set_ylabel('Amplitude')

tsample = np.arange(0, (30 * input), samp_time)
ssignal = np.sin(2 * np.pi * freq1 * tsample)

n = len(time)
noise_fft = np.fft.fft(resultsignal, n)
PSD = (np.abs(noise_fft) * np.abs(np.conj(noise_fft))) / n
Freq = (1 / (samp_time * n)) * np.arange(n)
L = np.arange(1, np.floor((n / 2)), dtype=int)

m5.stem(Freq[L], PSD[L], linefmt='g-', markerfmt='go')  
m5.set_title('PSD [Result Signal]')
m5.set_xlabel('Frequency in Hz')
m5.set_ylabel('Power Spectral Density [PSD]')

plt.tight_layout()
plt.show()
