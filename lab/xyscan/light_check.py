import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np


delta_t = (6.008 * 10 ** -7 - (-6.016 * 10 ** -7)) / 1504
t =  np.arange(-6.016 * 10 ** -7, 6.008 * 10 ** -7, delta_t)
# strnow  = '2022-01-31-1215_58_bared_PMT'

readname_list = ['./data/PMT/2022-01-31-1215_58_bared_PMT/', './data/PMT/2022-02-02-1647_48_bared_PMT/', './data/PMT/2022-01-30-1003_58_bared_PMT/', './data/PMT/2022-02-03-1308_15_bared_PMT/', './data/PMT/2022-01-30-1954_17_bared_PMT/', './data/PMT/2022-02-07-1928_02_gel_PMT/', './data/PMT/2022-02-08-1556_40_gel_PMT/', './data/PMT/2022-02-09-1724_26_gel_PMT/', './data/PMT/2022-02-10-1741_44_gel_PMT/', './data/PMT/2022-02-11-1654_21_gel_PMT/']      
n_list = ['light_check_start_ch1.txt', 'light_check_end_ch1.txt']

for readname in readname_list:
    a_list = []
    for n in n_list:
        data = np.loadtxt(readname + n, skiprows=1)
        data = -data
        charge_list = []
        # plt.figure()
        for k in range(len(data)):
            data[k] = data[k] - np.mean(data[k][:50])
            # plt.figure()
            # plt.plot(data[k])        
            value = integrate.simps(data[k][800:1100], t[800:1100]) 
            charge = (value / 50) * 10 ** 12
            charge_list.append(charge) 
        # print(np.mean(charge_list),'pc')
        a_list.append(np.mean(charge_list))
    print(a_list)
    
        # plt.show()
# print(np.mean(charge_list) / 1.60219 * 10 ** (-19) * 10 ** 12)