from logging import LogRecord
from time import time_ns
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import glob 


delt = 1.59999992 * 10 ** (-9) 
deltT = delt * 20000000
temp = -45

parameters = {'axes.labelsize': 16,
          'axes.titlesize': 18}
plt.rcParams.update(parameters)

def load_file(file):
    data = np.load(file)
    time = data['arr_0']
    wf = data['arr_1']
    return wf, time
def get_value(wf , time):
    wf = -wf
    wf = wf - np.mean(wf[:10])
    value1 = np.sum(wf[40:60] * delt) 
    charge = (value1 / 50) * 10 ** 12 / 10
    chg_wocut_list.append(charge)
    peak = np.max(wf) * 1000
    x = abs(max(wf)) / abs(min(wf))
    x_list.append(x)
    peak_list.append(peak)
    return x, peak, charge



file_list = glob.glob('./data/BB9759/' + f'{temp}' + '/darkrate/try2/*.npz')

chg_list = []
chg_wocut_list = []
x_all_list = []
peak_list = []
x_list = []
dt_list = []
plt.figure()
plt.title('waveform(noise)')
plt.xlabel('index delt = 2ns')
plt.ylabel('voltage[mV]')


for file in file_list:
    data = np.load(file)  
    time = data['arr_1']
    
    # plt.hist(time, bins=np.linspace(0, 2e7, 1000))
    # plt.plot(time)
    # plt.show()
    # print(time)
    # exit(1)
    index = []
    for i in range(len(data['arr_0'])):
        
        wf = data['arr_0'][i] 
        wf = -wf
        wf = wf - np.mean(wf[:10])
        # plt.plot(wf * 1000)
        value1 = np.sum(wf[40:60] * delt) 
        charge = (value1 / 50) * 10 ** 12 / 10
        chg_wocut_list.append(charge)
        peak = np.max(wf) * 1000
        x = abs(max(wf)) / abs(min(wf))
        x_list.append(x)

        peak_list.append(peak)
        if  (peak > 23.336 and x > 2.2):                                  # peak > 23.336 and x > 2.2 (-45degree) peak > 27 and x > 3.4 (-35degree) peak > 24 and x > 2.4 (-25 degree)  x < 3.4 and peak > 27 (-15 degree) peak > 24 and x > 2 (-5 degree)
            chg_list.append(charge)            
            plt.plot(wf * 1000)
            # print(i)
            index.append(i)
    
    time = [time[i] for i in index]
    for k in range(len(time) - 1):
        dt = (time[k + 1] - time[k]) * delt
        dt_list.append(dt)

# plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/noise_waveform.png')
plt.show()
plt.close()

dt_list = [np.log10(i) for i in dt_list]

plt.figure()
plt.yscale('log')
plt.title('deltT distribution')
plt.ylabel('events')
plt.xlabel('log10(E)')
plt.hist(dt_list, bins = 15)
plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/delt_distribution.png')
plt.show()

xbins = np.linspace(0, 40, 200)
ybins = np.linspace(0, 700, 200)
xc = 0.5*(xbins[1:]+xbins[:-1])
yc = 0.5*(ybins[1:]+ybins[:-1])
hist2d, binx, biny = np.histogram2d(x_list, peak_list, bins=[xbins, ybins])


plt.figure()
plt.yscale('log')
plt.title('peak height')
plt.xlabel('peak height[mV]')
plt.ylabel('events')
plt.hist(peak_list, bins = 100)
plt.tight_layout()
# plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/2dmap.png')
plt.show()
plt.close()



plt.figure()
plt.title('x vs peak height')
plt.xlabel('x')
plt.ylabel('peak height[mv]')
plt.pcolormesh(xc, yc, hist2d.T, norm = LogNorm(vmin=0.1, vmax=100))
plt.tight_layout()
plt.colorbar().set_label('')
# plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/2dmap.png')
plt.show()
plt.close()


rate = (len(chg_list)) / (deltT * 10 * 7)

plt.figure()
plt.title('charge histogram (no cutting)', fontsize = 18)
plt.yscale('log')
plt.ylabel('events')
plt.xlabel('charge[pc]')
plt.hist(chg_wocut_list, bins = 100, range = (np.min(chg_wocut_list), np.max(chg_wocut_list)))
# plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/charge_histogram_wo_cutting.png')
plt.show()
plt.close()

plt.figure()
plt.title('x=abs(max) / abs(min)')
plt.yscale('log')
plt.ylabel('events')
plt.xlabel('x')
plt.hist(x_list, bins = 100)
# plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/x_histogram.png')
plt.show()


plt.figure()
# plt.yscale('log')
plt.xlabel('charge[pc]')
plt.ylabel('events')
plt.title('charge histogram darkrate = ' + f'{rate:.2f}' + ' Hz')
plt.hist(chg_list, bins = 25)
plt.tight_layout()
# plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/charge_histogram_w_cutting.png')
plt.show()
plt.close()

print(deltT * 1000, 'ms')
print(len(chg_wocut_list))
print(len(chg_list))
print('darkrate = ', rate, ' Hz')

