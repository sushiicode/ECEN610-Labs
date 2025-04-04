import numpy as np
import matplotlib.pyplot as plt
import math

fig, (m1, m2, m3) = plt.subplots(3, 1)
plt.subplots_adjust(hspace=2.0)

freq = 1e9  
time = 1 / freq  
t = np.linspace(0, time, 100000)  

signal = np.sin(2 * np.pi * freq * t)
m1.plot(t, signal)
m1.set_xlabel('Time in s')
m1.set_ylabel('Amplitude')
m1.set_title('Original Signal Plot')

samp_Freq = 10e9 
samp_t = 1 / samp_Freq 
tsample = np.arange(0, time, samp_t)  
ss = np.sin(2 * np.pi * freq * tsample)  

m2.stem(tsample, ss, use_line_collection=True)
m2.set_xlabel('Time in s')
m2.set_ylabel('Amplitude')
m2.set_title('Sampled Signal Plot')

sample_hold = np.zeros(100000) 
sample_points = time / samp_t
points = 100000 / (2 * sample_points)
pulse = points - 1
signal_end = 0
count = 0

for k in range(int(sample_points)):
    if k != 0:
        pulse = k * (2 * points)

    for n in range(int(points)):
        time_inst = t[n]
        exponential = math.exp(-(time_inst / (10 * 10**-12)))  
        signal = ss[k]  
        sample_hold[count] = signal_end + (signal - signal_end) * (1 - exponential)
        count += 1

    for i in range(int(points)):
        signal_end = sample_hold[count - 1]
        sample_hold[count] = signal_end
        count += 1

m3.stem(t, sample_hold, use_line_collection=True)
m3.set_xlabel('Time in s')
m3.set_ylabel('Amplitude')
m3.set_title('Sample and Hold Response')
plt.show()
