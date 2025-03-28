#Q1.A

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

N = 8
Cs = 15.925e-12
freq = 2.4e9
fact = 1 / 2 * freq * Cs
num = fact * np.array([1, 1, 1, 1, 1, 1, 1, 1])
denom = fact * np.array([1])
freq, tf = freqz(num, denom)

plt.plot(freq, 20 * np.log10(abs(tf)), color='green')  # Set plot color to green
plt.grid()
plt.xlabel('Frequency in rad/sec')
plt.ylabel('Amplitude')
plt.title('Magnitude Response- Capacitor is Discharged')
plt.show()


#Q1b
Capacitors doesnt get discharged
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

N = 8
Cs = 15.925e-12
freq = 2.4e9
fact = 1 / 2 * freq * Csnum = fact * np.array([1, 1, 1, 1, 1, 1, 1, 1])
denom = fact * np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
freq, tf = freqz(num, denom)
plt.plot(freq, 20 * np.log10(abs(tf)), color='green')  # Set plot color to green
plt.grid()
plt.xlabel('Frequency in [rad/sec]')
plt.ylabel('Amplitude')
plt.title('Response when Capacitors are Not Discharged')
plt.show()





