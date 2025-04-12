#5a
import numpy as np
import matplotlib.pyplot as plt
dnl = np.array([0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0])
inl = np.zeros(8)
inl[0] = dnl[0]
for i in range(1, 8):
    inl[i] = inl[i - 1] + dnl[i]
code = np.arange(8)
plt.stem(code, inl, linefmt='b-', markerfmt='bx', basefmt='k')
plt.title('Integral Nonlinearity (INL)')
plt.xlabel('Code')
plt.ylabel('INL (LSB)')
plt.grid(True)
plt.show()
print('INL =', inl)


#5b
import matplotlib.pyplot as plt
import numpy as np
DNL = [0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0]
INL = np.cumsum(DNL)
code = np.arange(0, 8)
ideal = code + 0.5
ideal = ideal + 0.5
plt.plot(code, ideal, label='Ideal')
plt.plot(code, INL, label='ADC')
plt.xlabel('Code')
plt.ylabel('Voltage (LSB)')
plt.title('Transfer Curve')
plt.legend()
plt.show()
