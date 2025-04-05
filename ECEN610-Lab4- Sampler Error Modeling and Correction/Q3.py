#Q3.a

import numpy as np
import matplotlib.pyplot as plt
import math

fig, (m1, m2, m3, m4, m5) = plt.subplots(5, 1, figsize=(10, 12))

plt.xlabel('Time in S')
plt.ylabel('Amplitude')

f1 = 0.2e9
f2 = 0.58e9
f3 = 1e9
f4 = 1.7e9
f5 = 2.4e9
inputtime = 1 / f1
t = np.linspace(0, inputtime, 100000)

s1 = 0.1 * (np.sin(2 * np.pi * f1 * t))
s2 = 0.1 * (np.sin(2 * np.pi * f2 * t))
s3 = 0.1 * (np.sin(2 * np.pi * f3 * t))
s4 = 0.1 * (np.sin(2 * np.pi * f4 * t))
s5 = 0.1 * (np.sin(2 * np.pi * f5 * t))
signal = s1 + s2 + s3 + s4 + s5

m1.plot(t, signal, color='red')
m1.set_title('Multi-tone Original Signal Plot')

fs = 10e9
ts = 1 / fs
tsample = np.arange(ts / 2, inputtime, ts)

ss1 = 0.1 * (np.sin(2 * np.pi * f1 * tsample))
ss2 = 0.1 * (np.sin(2 * np.pi * f2 * tsample))
ss3 = 0.1 * (np.sin(2 * np.pi * f3 * tsample))
ss4 = 0.1 * (np.sin(2 * np.pi * f4 * tsample))
ss5 = 0.1 * (np.sin(2 * np.pi * f5 * tsample))

signalsample = ss1 + ss2 + ss3 + ss4 + ss5
m2.stem(tsample, signalsample)
m2.set_title('Sampled Multi-tone Signal Plot')

count = 0
samplehold = [0 for _ in range(100000)]
samplepts = inputtime / ts
points = 100000 / (2 * samplepts)
timeconst = 10.3e-12
pulse = points - 1
signalend = 0

for k in range(0, int(samplepts)):
    if k != 0:
        pulse = k * (2 * points)
    for n in range(0, int(points)):
        time = t[n]
        exp = math.exp(-(time / timeconst))
        sigtot = signal[int(pulse)]
        samplehold[count] = signalend + ((sigtot - signalend) * (1 - exp))
        count += 1
    for i in range(0, int(points)):
        signalend = samplehold[count - 1]
        samplehold[count] = signalend
        count += 1

m3.plot(t, samplehold)
m3.set_title('Practical Sample and Hold Signal Plot')

samp_error = [0 for _ in range(int(samplepts))]
for k in range(0, int(samplepts)):
    samp_error[k] = samplehold[int((k * 100000 / samplepts) + points)]

m4.stem(tsample, samp_error)
m4.set_title('Practical Sample and Hold Signal at Ts/2 Instant Plot')

error = np.array(samp_error) - signalsample
m5.plot(tsample, error)
m5.set_title('Error Signal Plot')

pwr = error ** 2
var = np.mean(pwr)
print('Variance=', var)

ratio = var * ((0.1 / 2 ** 7) ** 2 / 12)
print(ratio)

plt.tight_layout()
plt.show()

#3b

# Q3b
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2)
plt.subplots_adjust(left=None, bottom=None, right=2, top=2, wspace=None, hspace=None)

ref_adc_step_size = 1 / (2**12)
actual_step_size = 1 / (2**7)

freq1 = 0.2e9
freq2 = 0.58e9
freq3 = 1e9
freq4 = 1.7e9
freq5 = 2.4e9

inputtime = 1 / freq1
t = np.linspace(0, inputtime, 100000)

s1 = 0.1 * (np.sin(2 * np.pi * freq1 * t))
s2 = 0.1 * (np.sin(2 * np.pi * freq2 * t))
s3 = 0.1 * (np.sin(2 * np.pi * freq3 * t))
s4 = 0.1 * (np.sin(2 * np.pi * freq4 * t))
s5 = 0.1 * (np.sin(2 * np.pi * freq5 * t))

signal = s1 + s2 + s3 + s4 + s5

fs = 10e9
ts = 1 / fs
tsample = np.arange(ts / 2, inputtime, ts)

ss1 = 0.1 * (np.sin(2 * np.pi * freq1 * tsample))
ss2 = 0.1 * (np.sin(2 * np.pi * freq2 * tsample))
ss3 = 0.1 * (np.sin(2 * np.pi * freq3 * tsample))
ss4 = 0.1 * (np.sin(2 * np.pi * freq4 * tsample))
ss5 = 0.1 * (np.sin(2 * np.pi * freq5 * tsample))

signalsample = ss1 + ss2 + ss3 + ss4 + ss5
ax1.stem(tsample, signalsample, linefmt='green', markerfmt='go') 
ax1.set_xlabel('Time in S')
ax1.set_ylabel('Amplitude')
ax1.set_title('Sampled Signal Plot')

quantization_signal = np.round(signalsample / ref_adc_step_size)
quantized_signal = (quantization_signal * ref_adc_step_size)

ax2.stem(tsample, quantized_signal, linefmt='green', markerfmt='go')  
ax2.set_xlabel('Time in S')
ax2.set_ylabel('Amplitude')
ax2.set_title('Quantized Signal Plot')

plt.tight_layout()
plt.show()
