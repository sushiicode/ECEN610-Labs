#Q3a
# four capacitors discharged after their connection and read out option
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
# import numpy as np
#
# N = 32
# freq = 2.4e9
# C_History = 15.295e-12
# Cr = 0.5e-9
# a = C_History/C_History + Cr
# fact = 1/2 * freq * Cr
# num=fact *np.array([1,1,1,1,1,1,1,1])
# denom=fact *np.array([1,0,0,0,0,0,0,-a])
# freq,tf=freqz(num,denom)
# plt.plot(freq,20*np.log10(abs(tf)), color = 'green')
# plt.grid()
# plt.xlabel('Frequency in [rad/sec]')
# plt.ylabel('Amplitude')
# plt.title('Magnitude Response - 4 capacitors are discharged')
# plt.show()
