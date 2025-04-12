#Q3.a 

import matplotlib.pyplot as plt
import numpy as np
code = np.arange(8) 
ideal_op = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  
actual_op = [-0.01, 0.105, 0.195, 0.28, 0.37, 0.48, 0.6, 0.75] 
plt.figure(figsize=(8, 5))
plt.plot(code, ideal_op, 'b-o', label='Ideal Output (0.1 V/LSB)')
plt.plot(code, actual_op, 'r-x', label='Actual Output')
plt.scatter([0], [-0.01], color='purple', s=100, label='Offset Error (-0.1 LSB)')
plt.scatter([7], [0.75], color='green', s=100, label='Full-Scale Output (Error: 0.5 LSB)')
plt.xlabel('Code')
plt.ylabel('Output Voltage (V)')
plt.title('DAC Output: Offset and Full-Scale Error (Part a)')
plt.grid(True)
plt.legend()
plt.xticks(code)
plt.show()

#Q3.b

import matplotlib.pyplot as plt
import numpy as np
code = np.arange(8)
ideap_op = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # V
actual_op = [-0.01, 0.105, 0.195, 0.28, 0.37, 0.48, 0.6, 0.75]
ideal_gain_ln = [0.1 * k for k in code]
actual_gain_ln = [-0.01 + 0.108571 * k for k in code]
plt.figure(figsize=(8, 5))
plt.plot(code, ideap_op, 'b-o', label='Ideal Output (Gain: 1 LSB/code)')
plt.plot(code, actual_op, 'r-x', label='Actual Output (Gain: 1.08571 LSB/code)')
plt.plot(code, ideal_gain_ln, 'b--', alpha=0.5, label='Ideal Gain Line')
plt.plot(code, actual_gain_ln, 'r--', alpha=0.5, label='Actual Gain Line')
plt.xlabel('Code')
plt.ylabel('Output Voltage (V)')
plt.title('DAC Output: Ideal vs. Actual Gain (Part b)')
plt.grid(True)
plt.legend()
plt.xticks(code)
plt.show()

Q3.C

import matplotlib.pyplot as plt
import numpy as np
code = np.arange(8)  
dnl = [None, 0.05921, -0.17105, -0.21711, -0.17105, 0.01316, 0.10526, 0.38158]  
inl = [0, 0.05921, -0.11184, -0.32895, -0.5, -0.48684, -0.38158, 0]  
fig, (m1, m2) = plt.subplots(1, 2, figsize=(14, 5))
m1.bar(code[1:], dnl[1:], color='orange', edgecolor='black')
m1.set_xlabel('Code')
m1.set_ylabel('DNL (LSB)')
m1.set_title('Differential Nonlinearity (DNL) vs. Code (Part c)')
m1.grid(True, axis='y')
m1.set_xticks(code[1:])
m1.set_ylim(-0.4, 0.5)
m2.plot(code, inl, 'g-o', label='INL')
m2.set_xlabel('Code')
m2.set_ylabel('INL (LSB)')
m2.set_title('Integral Nonlinearity (INL) vs. Code (Part c)')
m2.grid(True)
m2.legend()
m2.set_xticks(code)
m2.set_ylim(-0.6, 0.2)
plt.tight_layout()
plt.show()

Q3.D

# Q3d
import numpy as np
import matplotlib.pyplot as plt

DAC_lvl = np.array([-0.01, 0.105, 0.195, 0.28, 0.37, 0.48, 0.6, 0.75])
ideal_lvl = np.arange(0, 0.8, 0.1)
code = np.arange(0, 8, 1)

offset_err = DAC_lvl[0] - ideal_lvl[0]
error_fs = DAC_lvl[7] - ideal_lvl[7]
dnl = np.zeros(8)
inl = np.zeros(8)
corr_lvl_dac = (DAC_lvl + 0.01) / 1.0857

inl[0] = dnl[0]
for i in range(1, 8):
    dnl[i] = (corr_lvl_dac[i] - corr_lvl_dac[i - 1] - 0.1) / 0.1
    inl[i] = inl[i - 1] + dnl[i]

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(code, dnl, linefmt='b-', markerfmt='bo', basefmt='k')
plt.xlabel('Code')
plt.ylabel('DNL')
plt.title('Differential Nonlinearity (DNL)')

plt.subplot(2, 1, 2)
plt.stem(code, inl, linefmt='g-', markerfmt='gx', basefmt='k')
plt.xlabel('Code')
plt.ylabel('INL')
plt.title('Integral Nonlinearity (INL)')

plt.tight_layout()
plt.show()

print('Actual =', DAC_lvl)
print('Ideal =', ideal_lvl)
print('Offset Error =', offset_err)
print('Full Scale Error =', error_fs)
print('Corrected =', corr_lvl_dac)
print('INL =', inl)
print('DNL =', dnl)
