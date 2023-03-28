from distutils import filelist
from logging import LogRecord
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import glob 
import function as func
from natsort import natsorted




parameters = {'axes.labelsize': 16,
          'axes.titlesize': 18}
plt.rcParams.update(parameters)

pmt = input('which pmt:') #BB9729_old, BB9783, BB9759
if pmt == 'BB9759':
    times = 3
    SPE = 78 # BB9759(-40) = 78mV, BB9729_old = 98.3mV, BB9783 = 111mV
elif pmt == 'BB9729_old':
    times = 3
    SPE = 98.3 # BB9759(-40) = 78mV, BB9729_old = 98.3mV, BB9783 = 111mV
elif pmt == 'BB9783':
    times = 1
    SPE = 111 # BB9759(-40) = 78mV, BB9729_old = 98.3mV, BB9783 = 111mV
else:
    print('exit')
    exit(1)
print(np.linspace(0.1, 0.5, 9)[4:5])
delt = 1.59999992 * 10 ** (-9) 
deltT = delt * 2e7
print('time width of 1 waveform =', deltT * 1000, 'ms')
temp = -40
thre_list = [SPE * i for i in np.linspace(0.1, 0.5, 9)]
print('0.3 SPE peak height = ', thre_list[4:5][0], 'mV')
print('threshold(mV) =', thre_list)

chg_list = []
chg_wocut_list = []
peak_list = []
x_list = []

dt_list = []
peak1_list = []
peak2_list = []
peak3_list = []
time = []
rate_list = []
t = np.linspace(-80, 80, 100)


graph_dir = './graph/' + pmt + f'/{temp}' + '/darkrate/'
file_list = glob.glob('./data/' + pmt + f'/{temp}' + '/darkrate/try' + str(times) + '/*.npz')
# file_list = glob.glob('./data/' + pmt + f'/{temp}' + '/darkrate/take/*.npz')
file_list = natsorted(file_list)
print('number of file is', len(file_list))                             

i = 0
for thre in thre_list[4:5]:

    chg_list = []
    chg_wocut_list = []
    peak_list = []
    x_list = []
    dt_list = []
    peak1_list = []
    peak2_list = []
    peak3_list = []
    time = []
    i = 0
    t = np.linspace(-80, 80, 100)
    # plt.figure()
    # plt.title('waveform')
    # plt.ylabel('voltage[mV]')
    # plt.xlabel('time[ns]')
    for file in tqdm(file_list):
        # print(file)
        data = np.load(file)  
        i += 1
        # time = data['arr_1']
        ped = data['arr_2']
        index = []
        for iwf, wf in enumerate(data['arr_0']):
            # hist, bins = np.histogram(wf, bins = 100)
            # x1 = (bins[1:] + bins[:-1]) / 2
            # print(x)
            # ped = x1[np.argmax(hist)]
            # print(ped)
            wf = wf - ped
            wf = -wf
            # F = np.fft.fft(wf) # 変換結果
            # F_abs = np.abs(F)
            # plt.plot(F_abs)
            # plt.show()
            # plt.figure()
            # plt.title('waveform')
            # plt.ylabel('voltage[mV]')
            # plt.xlabel('time[ns]')
            # plt.plot(t, wf * 1000)
            # plt.show()
            value1 = np.sum(wf[40:60] * delt) 
            charge = (value1 / 50) * 10 ** 12 / 10
            chg_wocut_list.append(charge)
            peak = np.max(wf[40:60]) * 1000
            x = abs(max(wf)) / abs(min(wf))
            x_list.append(x)
            peak_list.append(peak)

            if (peak > thre and x > 3.4):     
                # peak_list.append(peak)
                # plt.figure()
                # plt.title('waveform')
                # plt.ylabel('voltage[mV]')
                # plt.xlabel('time[ns]')
                # plt.plot(t, wf * 1000)
                # plt.show()  
                time.append(i)
                chg_list.append(charge)    
                if 0 < i < 300:
                    peak1_list.append(peak)
                if 300 < i < 600:
                    peak2_list.append(peak)
                if 600 < i < 900:
                    peak3_list.append(peak)
                
        # time = [time[i] for i in index]
        # for k in range(len(time) - 1):
        #     dt = (time[k + 1] - time[k]) * delt
        #     dt_list.append(dt)
    xbins = np.linspace(0, 20, 100)
    ybins = np.linspace(0, 300, 100)
    xc = 0.5*(xbins[1:] + xbins[:-1])
    yc = 0.5*(ybins[1:] + ybins[:-1])
    hist2d, binx, biny = np.histogram2d(x_list, peak_list, bins=[xbins, ybins])

    plt.figure()
    plt.title('x vs peak height')
    plt.xticks([3.4])
    plt.yticks([thre])
    plt.xlabel('x')
    plt.ylabel('peak height[mv]')
    plt.axvline(x = 3.4)
    plt.axhline(y = thre)
    plt.pcolormesh(xc, yc, hist2d.T, norm = LogNorm(vmin=0.1, vmax=100))
    plt.tight_layout()
    plt.colorbar().set_label('')
    # plt.savefig(graph_dir + '2dmap_try' + str(times) + '.png')
    plt.show()
    plt.close()


    rate = (len(chg_list)) / (deltT  * len(file_list))
    rate_list.append(rate)
    print(len(chg_wocut_list))
    print('# = ', len(chg_list))
    print('darkrate = ', rate, ' Hz')

    # plt.tight_layout() 
    # plt.show()
    # plt.close()


# plt.figure()                                                                                                                                                                                                                                                                                                                                                                                                                                                
# plt.yscale('log')
# plt.xlabel('800waveforms(1waveform is 32 ms)', fontsize = 16)                                                                                                                                                                                                                                                                                                                                                                           
# plt.ylabel('events')
# plt.hist(time, bins = np.linspace(-0.5, 800.5, 801))
# plt.title('time')
# # plt.savefig(graph_dir + 'timing_distributions_try' + str(times) + '.png')
# plt.show()
# plt.close()

# bin = np.linspace(0, 150, 50)
# plt.title('peak')
# plt.hist(peak1_list, bins = bin, histtype= 'step', label = '0 < times < 300')
# plt.hist(peak2_list, bins = bin,  histtype= 'step', label = '300 < times < 600')
# plt.hist(peak3_list, bins = bin, histtype= 'step', label = '600 < times < 900')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
# plt.legend()
# # plt.savefig(graph_dir + 'timing_peak_distributions_try' + str(times) + '.png')
# plt.show()
# plt.close()


# plt.figure()
# plt.yscale('log')
# plt.title('deltT distribution')
# plt.ylabel('events')
# plt.xlabel('log10(E)')
# plt.hist(dt_list, bins = 15)
# # plt.savefig('./graph/BB9759/' + f'{temp}' + '/darkrate/delt_distribution_try' + str(times) + '.png')
# plt.show()



# F = np.fft.fft(peak_list) # 変換結果
# freq = np.fft.fftfreq(len(peak_list), d = 11 / (160 * 1e-9)) # 周波数

# fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
# ax[0].plot(F.real, label="Real part")
# ax[0].legend()
# ax[1].plot(F.imag, label="Imaginary part")
# ax[1].legend()
# ax[2].plot(freq, label="Frequency")
# ax[2].legend()
# ax[2].set_xlabel("Number of data")
# plt.show()


# nbin = 400
# F = func.Fitting(peak_list)
# popt_peak = F.Fit_gaussian_1peak(nbin, 15, 95, 20)
# x_peak = np.linspace(10, 200, 100)
# y_peak = F.Gaussian_func_1peak(x_peak, popt_peak[0], popt_peak[1], popt_peak[2])
# print('thre = ',popt_peak[1], 'mV')


plt.figure()
plt.yscale('log')
plt.title('peak height')
plt.xlabel('peak height[mV]')
plt.ylabel('events')
plt.hist(peak_list, bins = np.linspace(10, 200, 100))
# plt.plot(x_peak, y_peak)
# plt.axvline(x = popt_peak[1], color = 'black', label = 'SPE peak = ' + '{:.1f} mV'.format(popt_peak[1]))
plt.tight_layout()
plt.legend()
# plt.savefig(graph_dir + 'after_peakcutting_peak_height_try' + str(times) + '.png')
plt.show()
plt.close()




# plt.figure()
# plt.title('charge histogram (no cutting)', fontsize = 18)
# plt.yscale('log')
# plt.ylabel('events')
# plt.xlabel('charge[pc]')
# plt.hist(chg_wocut_list, bins = 100)
# # plt.savefig(graph_dir + 'charge_histogram_wo_cutting.png')
# plt.show()
# plt.close()


# plt.figure()
# plt.title('x=abs(max) / abs(min)')
# plt.yscale('log')
# plt.ylabel('events')
# plt.xlabel('x')
# plt.axvline(x = 3.4)
# plt.hist(x_list, bins = np.linspace(0, 5, 50))
# # plt.savefig(graph_dir + 'x_histogram_try' + str(times) + '.png')
# plt.show()


plt.figure()
plt.yscale('log')
plt.xlabel('charge[pc]')
plt.ylabel('events')
plt.title('charge histogram darkrate = ' + f'{rate:.2f}' + ' Hz  (cut by 0.3 SPE peak height)')
plt.hist(chg_list, bins = np.linspace(0, 5, 80))
plt.tight_layout()
# plt.savefig(graph_dir + 'charge_histogram_after_x_and_peakcut_try' + str(times) + '.png')
plt.show()
plt.close()

print(rate_list)