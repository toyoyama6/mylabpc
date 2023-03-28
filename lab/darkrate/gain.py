import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit  
from tqdm import tqdm
import function as func



pmt = 'BB9729_old'
gain_list = []
volt_list = [80.0, 82.8, 88.4, 91.2, 94.0]
# volt_list = [94.0]
# t_start = -5.892 * 10 ** (-7)
# t_end = 6.104 * 10 ** (-7)
t_start = -4.8 * 10 ** (-7)
t_end = 7.198 * 10 ** (-7)

deltat = (t_end - t_start) / 6000
t =  np.arange(t_start, t_end, deltat)
temp = -40
nbin = 100


for volt in tqdm(volt_list):
    readname = './data/' + pmt + f'/{temp}' + '/gain/GainMes_V' + str(volt) + '_ch2.txt'
    data = np.loadtxt(readname, skiprows = 1)
    charge_list, peak_list = func.Value().Get_charge_and_peak(data, t, 2950, 3180)
    # peak_list = [i * 10 for i in peak_list]
    F = func.Fitting(charge_list)
    popt_chg = F.Fit_gaussian_2peak(nbin, 0.2, 20, 1, 0.2)
    # popt_peak = F.Fit_gaussian_2peak(nbin, 2, 110, 60, 30)
    x_chg = np.linspace(np.min(charge_list), np.max(charge_list), 1000)
    # x_peak = np.linspace(np.min(peak_list), np.max(peak_list), 1000)
    y_chg = F.Gaussian_func_2peak(x_chg, popt_chg[0], popt_chg[1], popt_chg[2], popt_chg[3], popt_chg[4], popt_chg[5])
    # y_peak = F.Gaussian_func(x_peak, popt_peak[0], popt_peak[1], popt_peak[2], popt_peak[3], popt_peak[4], popt_peak[5])


    plt.figure()
    plt.yscale('log')
    plt.ylabel('events', fontsize = 16)
    plt.xlabel('charge', fontsize = 16)
    plt.axvline(x = popt_chg[4])
    plt.hist(charge_list, bins = nbin)
    plt.plot(x_chg, y_chg)
    plt.title('volt = ' + str(12 * volt), fontsize = 18)
    plt.tight_layout()
    # plt.savefig('./graph/' + pmt + f'/{temp}/' + f'BB9759 temperture = {temp} volt = ' + str(12 * volt) + '.png')
    plt.show() 
    plt.close()
    gain_list.append((popt_chg[4] - popt_chg[1]) / (1.60219 * 10 ** (-19) * 10 ** 12))

    # plt.figure()
    # plt.yscale('log')
    # plt.ylabel('events', fontsize = 16)
    # plt.xlabel('peak[mV]', fontsize = 16)
    # plt.axvline(x = 25, color = 'black')
    # plt.hist(peak_list * 10, bins = np.linspace(0, 200, 100))
    # # plt.plot(x_peak,y_peak)
    # plt.title('volt = ' + str(12 * volt), fontsize = 18)
    # plt.tight_layout()
    # # plt.savefig('./detect_peak.png')
    # # plt.savefig('./graph/' + pmt + f'/{temp}/' + f'BB9759 temperture = {temp} volt = ' + str(12 * volt) + '.png')
    # plt.show() 
    # plt.close()


volt_list = [x * 12 for x in volt_list]
a, b = np.polyfit(volt_list, np.log10(gain_list), 1)
y = [a * v + b for v in volt_list]

volt = (np.log10(5 * 10 ** 6) - b) / a

plt.figure()
plt.scatter(volt_list, np.log10(gain_list))
plt.scatter((np.log10(5 * 10 ** 6) - b) / a, np.log10(5 * 10 ** 6), marker = '*', color = 'red', s = 300)
plt.plot(volt_list, y, label = f'{volt}')
plt.xlabel('volt', fontsize = 16)
plt.ylabel('log10(gain)', fontsize = 16)
plt.title(f'temperture={temp}', fontsize = 18)
plt.legend()
plt.tight_layout()
# plt.savefig('./graph/' + pmt + f'/{temp}' + '/gain_fit' + f'temperture={temp}.png')
plt.show()
plt.close()

# print(volt_list)
# print(np.log10(gain_list))
