import numpy as np
import matplotlib.pyplot as plt
import math

fig, (m1, m2, m3) = plt.subplots(3)
fig.subplots_adjust(hspace=2.0)

f1 = 0.2e9
f2 = 0.58e9
f3 = 1e9
f4 = 1.7e9
f5 = 2.4e9

input_time = 1 / f1
t = np.linspace(0, input_time, 100000)

sig1 = 0.1 * np.sin(2 * np.pi * f1 * t)
sig2 = 0.1 * np.sin(2 * np.pi * f2 * t)
sig3 = 0.1 * np.sin(2 * np.pi * f3 * t)
sig4 = 0.1 * np.sin(2 * np.pi * f4 * t)
sig5 = 0.1 * np.sin(2 * np.pi * f5 * t)
signal = sig1 + sig2 + sig3 + sig4 + sig5

m1.plot(t, signal, color='red')
m1.set_xlabel('Time in s')
m1.set_ylabel('Amplitude')
m1.set_title('Multitone Original Signal')

sampling_freq = 10e9
samp_time = 1 / sampling_freq
tsample = np.arange(samp_time / 2, input_time, samp_time)

ss1 = 0.1 * np.sin(2 * np.pi * f1 * tsample)
ss2 = 0.1 * np.sin(2 * np.pi * f2 * tsample)
ss3 = 0.1 * np.sin(2 * np.pi * f3 * tsample)
ss4 = 0.1 * np.sin(2 * np.pi * f4 * tsample)
ss5 = 0.1 * np.sin(2 * np.pi * f5 * tsample)
signal_sample = ss1 + ss2 + ss3 + ss4 + ss5

m2.stem(tsample, signal_sample, use_line_collection=True)
m2.set_xlabel('Time in s')
m2.set_ylabel('Amplitude')
m2.set_title('Ideal Sampled Multitone Signal Plot')

count = 0
samplehold = [0 for _ in range(100000)]
samplepoints = input_time / samp_time
points = int(100000 / (2 * samplepoints))
pulse = points - 1
signalend = 0
timeconst = 10.3e-12

for k in range(int(samplepoints)):
    if k != 0:
        pulse = int(k * (2 * points))
    for n in range(points):
        time = t[n]
        exponential = math.exp(-time / timeconst)
        signaltot = signal[int(pulse)]
        samplehold[count] = signalend + (signaltot - signalend) * (1 - exponential)
        count += 1
    for i in range(points):
        signalend = samplehold[count - 1]
        samplehold[count] = signalend
        count += 1

m3.plot(t, samplehold)
m3.set_xlabel('Time in s')
m3.set_ylabel('Amplitude')
m3.set_title('Practical Sample and Hold Output Response')

plt.show()

