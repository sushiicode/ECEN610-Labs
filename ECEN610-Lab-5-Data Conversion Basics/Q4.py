# Q4 - DNL and INL from Ramp Histogram for 4-bit ADC

import numpy as np
import matplotlib.pyplot as plt

Histadc = np.array([43, 115, 85, 101, 122, 170, 75, 146,
                    125, 60, 95, 95, 115, 40, 120, 242])
tot = np.sum(Histadc)
lsb = tot / 16
code = np.arange(16)

dnl = np.zeros(16)
inl = np.zeros(16)

for i in range(16):
    dnl[i] = (Histadc[i] - lsb) / lsb

inl[0] = dnl[0]
for i in range(1, 16):
    inl[i] = inl[i - 1] + dnl[i]

fig, m1 = plt.subplots(2, 1, figsize=(10, 6))
m1[0].stem(code, dnl, linefmt='r-', markerfmt='ro', basefmt='k')
m1[0].set_title('Differential Nonlinearity (DNL)')
m1[0].set_xlabel('Code')
m1[0].set_ylabel('DNL (LSB)')
m1[0].grid(True)
m1[1].stem(code, inl, linefmt='g-', markerfmt='gx', basefmt='k')
m1[1].set_title('Integral Nonlinearity (INL)')
m1[1].set_xlabel('Code')
m1[1].set_ylabel('INL (LSB)')
m1[1].grid(True)
plt.tight_layout()
plt.show()
print('Total Samples =', tot)
print('Ideal LSB Count =', lsb)
print('DNL =', dnl)
print('INL =', inl)
