import matplotlib.pyplot as plt
import numpy as np

delt = 1.59999992 * 10 ** (-9) 
deltT = delt * 20000000
readname = './rate.txt'
data = np.loadtxt(readname, skiprows = 1)
rate_list = []
temp_list = []

for i in range(len(data)):
    rate_list.append(data[i][0])
    temp_list.append(data[i][1])

number_list = [i * deltT * 10 * 7 for i in rate_list]
error_list = [np.sqrt(i) for i in number_list]
print(number_list)
print(temp_list)
print(error_list)

plt.figure()
# plt.grid()
plt.yscale('log')
plt.title('rate vs temperture', fontsize = 18)
plt.ylabel('rate[Hz]', fontsize = 16)
plt.xlabel('degree', fontsize = 16)
plt.errorbar(temp_list, rate_list, yerr = error_list, capsize=5, fmt='o', markersize=10, ecolor='blue', markeredgecolor = "blue", color='w')
# plt.savefig('./graph/BB9759/rate_temp.png')
plt.tight_layout()
plt.show()
