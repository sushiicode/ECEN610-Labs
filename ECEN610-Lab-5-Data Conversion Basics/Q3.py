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
