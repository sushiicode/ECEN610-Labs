#Q1 Python Code

#Q1.A

# FIR Filter Response


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

num=[0.5,0.5,0.5,0.5]
den=[1]
w, h = freqz(num,den)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Magnitude response (FIR filter)')
plt.xlabel('Frequency in rad/sample')
plt.ylabel('Magnitude in dB')
plt.grid(True)
plt.show()


# IIR Filter Response

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

a=[0.7,0.7]
b=[1,-0.7]

A, B = freqz(a,b)
plt.plot(A, 20 * np.log10(abs(B)))
plt.xscale('log')
plt.title('Magnitude response of the IIR filter')
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.show()


#Q1.B

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

num=[1,1,1,1,1]
den=[1]
a, b = freqz(num,den)
plt.plot(a, 20 * np.log10(abs(b)))
plt.xscale('log')
plt.title('Magnitude response of the FIR filter')
plt.xlabel('Frequency [rad/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)
plt.show()

#Modified code for finding zeros and poles

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk

a = [1, 1,1,1,1]
b = [1]
A, B = freqz(a, b)
plt.plot(A, 20 * np.log10(abs(B)))
plt.xscale('log')
plt.title('Magnitude response (FIR filters) ')
plt.xlabel('Frequency in rad/sample')
plt.ylabel('Magnitude in dB')
plt.grid(True)
plt.show()
zeros, poles, gain = tf2zpk(a,b)
print("Zeros ( FIR filter) :")
print(zeros)
print("Poles (FIR filter) :")
print(poles)









