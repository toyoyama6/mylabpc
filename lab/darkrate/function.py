import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit  
import math
from tqdm import tqdm

class Fitting:
    def __init__(self, x_list):
        self.x_list = x_list
    
    def Gaussian_func_2peak(self, x, Aped, Mped, Sped, Aspe, Mspe, Sspe):
            return Aped * np.exp(-(x - Mped) ** 2 / (2 * Sped ** 2)) + Aspe * np.exp(-(x - Mspe) ** 2 / (2 * Sspe ** 2))

    def Fit_gaussian_2peak(self, bin, Sped, Aspe, Mspe, Sspe):    
        hist, bins = np.histogram(self.x_list, bins = bin, range=(np.min(self.x_list), np.max(self.x_list)))
        class_value_list = []
        for k in range(len(bins) - 1):
            class_value = (bins[k] + bins[k + 1]) / 2
            class_value_list.append(class_value)

        popt, pcov = curve_fit(self.Gaussian_func_2peak, class_value_list, hist, p0 = [np.amax(hist), class_value_list[np.argmax(hist)], Sped, Aspe, Mspe, Sspe])
        return popt 

    def Gaussian_func_1peak(self, x, Aped, Mped, Sped):
            return Aped * np.exp(-(x - Mped) ** 2 / (2 * Sped ** 2))

    def Fit_gaussian_1peak(self, bin, Aped, Mped, Sped):    
        hist, bins = np.histogram(self.x_list, bins = bin, range=(np.min(self.x_list), np.max(self.x_list)))
        class_value_list = []
        for k in range(len(bins) - 1):
            class_value = (bins[k] + bins[k + 1]) / 2
            class_value_list.append(class_value)

        popt, pcov = curve_fit(self.Gaussian_func_1peak, class_value_list, hist, p0 = [Aped, Mped, Sped])
        return popt 


class Value:  
    def Get_charge_and_peak(self, data, t, start, end):
        charge_list = []
        peak_list = []
        for wf in data:	
            wf = -wf 
            wf = wf - np.mean(wf[1000:2000])
            # plt.plot(wf)
            peak = np.max(wf[start:end])
            charge = integrate.simps(wf[start:end], t[start:end]) / 10000 / 50 * 1e12
            charge_list.append(charge)
            peak_list.append(peak)
        # plt.show()
        return charge_list, peak_list

class Plot:
    def Get_histogram(self, data,  start, end, bins, title, ylabel, xlabel, graph_dir):
        plt.figure()
        plt.title(title, fontsize = 18)
        plt.ylabel(ylabel, fontsize = 16)
        plt.xlabel(xlabel,  fontsize = 16)
        plt.hist(data, bins = np.linspace(start, end, bins))
        plt.savefig(graph_dir)
        plt.show()
        plt.close()

    def Get_plot(self, x, y, title, ylabel, xlabel, graph_dir):
        plt.figure()
        plt.title(title, fontsize = 18)
        plt.ylabel(ylabel, fontsize = 16)
        plt.xlabel(xlabel,  fontsize = 16)
        plt.plot(x, y)
        plt.savefig(graph_dir)
        plt.show()
        plt.close()

    def Get_scattering(self, x, y, title, ylabel, xlabel, graph_dir):
        plt.figure()
        plt.title(title, fontsize = 18)
        plt.ylabel(ylabel, fontsize = 16)
        plt.xlabel(xlabel,  fontsize = 16)
        plt.plot(x, y)
        plt.savefig(graph_dir)
        plt.show()
        plt.close()
