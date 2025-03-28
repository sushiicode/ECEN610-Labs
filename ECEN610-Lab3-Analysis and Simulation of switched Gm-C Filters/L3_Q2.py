#Q2a
#Effect of adding CH capacitor

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

N = 8
C_History= 15.425e-12
Cr = 0.5e-12
freq = 2.4e9
factor=1/2*freq*Cr
a=C_History/C_History+Cr
numerator=factor*np.array([1,1,1,1,1,1,1,1])
denominator=factor*np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-a])
freq,tf=freqz(numerator,denominator)
plt.plot(freq,20*np.log10(abs(tf)), color = 'green')
plt.grid()
plt.xlabel('Frequency in [rad/sec]')
plt.ylabel('Amplitude')
plt.title('Magnitude Response with History Capacitor')
plt.show()
