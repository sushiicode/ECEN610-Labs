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
