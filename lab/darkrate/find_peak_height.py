import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from scipy.stats import norm
from scipy import integrate
from tqdm import tqdm
from scipy.optimize import curve_fit  
import math
from scipy.signal import argrelmin, argrelmax
from scipy import signal
import function as func


 
t_start = -6 * 10 ** (-7)
t_end = 5.998 * 10 ** (-7)
deltat = (t_end - t_start) / 6000
t =  np.arange(t_start, t_end, deltat)




readname = './data/BB9783/-40/darkrate/peak_height.txt'
data = np.loadtxt(readname, skiprows = 1)
charge_list, peak_list = func.Value().Get_charge_and_peak(data, t, 3350, 3650)

nbin = 300
F = func.Fitting(peak_list)
popt_peak = F.Fit_gaussian_1peak(400, 130, 120, 60)
x_peak = np.linspace(50, 250, nbin)
y_peak = F.Gaussian_func_1peak(x_peak, popt_peak[0], popt_peak[1], popt_peak[2])
print(popt_peak[1], 'mV')


plt.figure()
plt.yscale('log')
plt.title('peak height')
plt.xlabel('peak height[mV]')
plt.ylabel('events')
plt.hist(peak_list, bins = np.linspace(30, 300, 200))
plt.plot(x_peak, y_peak)
plt.axvline(x = popt_peak[1], color = 'black', label = 'SPE peak = ' + '{:.1f} mV'.format(popt_peak[1]))
plt.tight_layout()
plt.legend()
# plt.savefig(graph_dir + 'after_peakcutting_peak_height_try' + str(times) + '.png')
plt.show()
plt.close()


