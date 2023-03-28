from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from tqdm import tqdm



parameters = {'axes.labelsize': 25,
          'axes.titlesize': 25}
plt.rcParams.update(parameters)


graph_dir = './graph/'
volt_list = np.linspace(80, 94, 6)
t_start = -3.068 * 10 ** (-7)	
t_end = 5.952 * 10 ** (-7)
deltat = (t_end - t_start) / 2256
t =  np.arange(t_start, t_end, deltat)
nbin = 100

def Get_charge_and_peak(data, t, start, end):
        charge_list = []
        peak_list = []
        for wf in data:
            wf = -wf
            wf = wf - np.mean(wf[:500])
			#plt.ylabel('amplitude [mV]', fontsize = 25)
            #plt.xlabel('time [ns]', fontsize = 25)
            # plt.xlim(80,180)
            # plt.plot(wf)
            #plt.tight_layout()
            # plt.show()
            charge = integrate.simps(wf[start:end], t[start:end]) / 1000 / 50 * 1e12
            charge_list.append(charge)
        # plt.show()
        return charge_list
    
def Gaussian_func_2peak(x, Aped, Mped, Sped, Aspe, Mspe, Sspe):
    return Aped * np.exp(-(x - Mped) ** 2 / (2 * Sped ** 2)) + Aspe * np.exp(-(x - Mspe) ** 2 / (2 * Sspe ** 2))

def Fit_gaussian_2peak(charge_list, bin, Aped, Mped, Sped, Aspe, Mspe, Sspe):
    hist, bins = np.histogram(charge_list, bins = bin, range=(np.min(charge_list), np.max(charge_list)))
    class_value_list = []
    for k in range(len(bins) - 1):
        class_value = (bins[k] + bins[k + 1]) / 2
        class_value_list.append(class_value)
    popt, pcov = curve_fit(Gaussian_func_2peak, class_value_list, hist, p0 = [Aped, Mped, Sped, Aspe, Mspe, Sspe])
    return popt

flag = input('fit or no: ')

gain_list = []
for volt in tqdm(volt_list):
    readname = './data/GainMes_V' + str(volt) + '_ch1.txt'
    data = np.loadtxt(readname, skiprows = 1)
    charge_list = Get_charge_and_peak(data, t, 1020, 1120)
    plt.figure()
    plt.yscale('log')
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.ylim(0.8, 500)
    plt.ylabel('events')
    plt.xlabel('charge [pC]')
    plt.hist(charge_list, bins = nbin, histtype = 'step', color = 'blue')
    plt.title('volt = ' + '{:.1f}'.format(12 * volt) +  ' [V]')
    plt.tight_layout()
    
    if flag == 'fit':
        popt = Fit_gaussian_2peak(charge_list, nbin, 200, 0, 0.2, 80, 1, 0.3)
        x_chg = np.linspace(np.min(charge_list), np.max(charge_list), 1000)
        y_chg = Gaussian_func_2peak(x_chg, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        plt.axvline(x = popt[4], color = 'black')
        plt.plot(x_chg, y_chg, c = 'red')
        gain_list.append(popt[4] / (1.60219 * 10 ** (-19) * 10 ** 12))


    plt.savefig(graph_dir + f'fitting_volt=' + '{:.0f}'.format(12 * volt) + '.png')
    #plt.show()
    plt.close()


volt_list = [x * 12 for x in volt_list]
a, b = np.polyfit(volt_list, np.log10(gain_list), 1)
y = [a * v + b for v in volt_list]
#x = np.linspace()
volt = (np.log10(5 * 10 ** 6) - b) / a
print(volt)
plt.figure()
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.scatter(volt_list, np.log10(gain_list))
plt.scatter((np.log10(5 * 10 ** 6) - b) / a, np.log10(5 * 10 ** 6), marker = '*', color = 'red', s = 300)
plt.plot(volt_list, y)
plt.axvline(x = volt, ymin = 0, ymax = np.log10(5 * 10 ** 6), linestyle = 'dashed')
plt.xlabel('volt [V]')
plt.ylabel('log10(gain)')
plt.tight_layout()
plt.savefig(graph_dir + 'gain_fit.png')
#plt.show()
plt.close()
print('volt', (np.log10(5 * 10 ** 6) - b) / a, np.log10(5 * 10 ** 6))

