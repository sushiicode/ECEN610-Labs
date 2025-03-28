#Q3a

#four capacitors discharged
import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np

N = 32
freq = 2.4e9
C_History = 15.425e-12
Cr = 0.5e-12
a = C_History/C_History + Cr
fact = 1/2 * freq * Cr
num=fact *np.array([1,1,1,1,1,1,1,1])
denom=fact *np.array([1,0,0,0,0,0,0,-a])
freq,tf=freqz(num,denom)
plt.plot(freq,20*np.log10(abs(tf)), color = 'green')
plt.grid()
plt.xlabel('Frequency in [rad/sec]')
plt.ylabel('Amplitude')
plt.title('Magnitude Response - 4 capacitors are discharged')
plt.show()

#Q3b

# four capacitors are never discharged
import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np
N=32
Freq = 2.4e9
C_history =15.295e-12
Cr = 0.5e-9
a= C_history /C_history + Cr
fact =1/2* Freq * Cr
num = fact *np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
denominator=fact *np.array([1,0,0,0,0,0,0,0,-a])
freq,tf=freqz(num,denominator)
plt.plot(freq,20*np.log10(abs(tf)), color = 'green')
plt.grid()
plt.xlabel('Frequency in [rad/sec]')
plt.ylabel('Amplitude')
plt.title('Magnitude Response - Capacitors are Never Discharged')
plt.show()


#Q3c

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math

N = 8
f_c = 2.4e9
Cs = 15.925e-12
DT = 1 / (2 * f_c)
C_History = 15.425e-12
Cr = 0.5e-12
a = (C_History / (C_History + Cr))

print("a ", a)
print("1-a", 1 - a)

cr1 = Cr
cr2 = 2 * Cr
cr3 = 3 * Cr
cr4 = 4 * Cr

m1 = (C_History / (C_History + cr1))
m2 = (C_History / (C_History + cr2))
m3 = (C_History / (C_History + cr3))
m4 = (C_History / (C_History + cr4))

x = (DT) / Cs
fir1 = x * np.array([1, 1, 1, 1, 1, 1, 1, 1])
fir2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, (1 - a)])
fir3 = np.array([m1, 0, 0, 0, 0, 0, 0, 0, (m1 * m2), 0, 0, 0, 0, 0, 0, 0, (m1 * m2 * m3), 0, 0, 0, 0, 0, 0, 0, (m1 * m2 * m3 * m4)])
iir1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, -a])
iir2 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
m1, TF_1 = signal.freqz(fir1, 1, worN=80001, whole=False)
m2, TF_2 = signal.freqz(fir2, 1, worN=80001, whole=False)
m3, TF_3 = signal.freqz(fir3, 1, worN=80001, whole=False)
m4, TF_4 = signal.freqz(1, iir1, worN=80001, whole=False)
m5, TF_5 = signal.freqz(1, iir2, worN=80001, whole=False)
plt.figure()
plt.plot(((f_c * m1) / (2 * np.pi)), 10 * np.log10(np.abs(TF_1)) + 10 * np.log10(np.abs(TF_2)) + 10 * np.log10(np.abs(TF_3)) + 10 * np.log10(np.abs(TF_4)) + 10 * np.log10(np.abs(TF_5)), color='green')
plt.grid()
plt.xlabel("Angular Freq")
plt.ylabel("Transfer Func")
plt.show()
