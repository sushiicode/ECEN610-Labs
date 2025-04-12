#5a
import matplotlib.pyplot as plt
import numpy as np
DNL = [0, -0.5, 0, 0.5, -1, 0.5, 0.5, 0]
INL = np.cumsum(DNL)
code = np.arange(0, 8)
ideal = code + 0.5
ideal = ideal + 0.5
plt.plot(code, ideal, label='Ideal')
plt.plot(code, INL, label='ADC', color='green')
plt.xlabel('Code')
plt.ylabel('Voltage (LSB)')
plt.title('Transfer Curve')
plt.legend()
plt.grid(True)
plt.show()


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
