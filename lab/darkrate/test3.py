import numpy as np 
import matplotlib.pyplot as plt   


a = np.array([89, 34, 56, 87, 90, 23, 45, 12, 65, 78, 9, 34, 12, 11, 2, 65, 78, 82, 28, 78]) 

histogram = np.histogram(a) 
print(histogram) 
plt.hist(a)
plt.show()