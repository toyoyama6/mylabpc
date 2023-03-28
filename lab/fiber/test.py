
import numpy as np



a = np.linspace(2.5, 2.8, 30)
print(a)

v_list = np.linspace(2.5, 2.8, 30)

print(v_list)

v_list = [float(f'{i:.2f}') for i in v_list]


print(v_list)