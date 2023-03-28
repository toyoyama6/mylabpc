import pandas as pd 
import glob
from pathlib import Path
from pylab import *
import csv
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

data_dir = './gaindata/'
graph_dir = './gain_graph/'
bin = 200
int_x = int(4000/bin)

data_d = sys.argv[1]


file_name = sys.argv[1] + '_gain-plot_'

# data=concatenate((normal(1,.2,5000),normal(2,.2,2500)))

# print(data)

df = glob.glob(data_dir+data_d+'/'+'G*.txt')

def func(x, *params):

    #paramsの長さでフィッティングする関数の数を判別。
    num_func = int(len(params)/3)

    #ガウス関数にそれぞれのパラメータを挿入してy_listに追加。
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)

    #y_listに入っているすべてのガウス関数を重ね合わせる。
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i

    #最後にバックグラウンドを追加。
    y_sum = y_sum + params[-1]

    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2) + params[-1]
        y_list.append(y)
    return y_list

d = [1500, 850, 1000, 900, 900, 1000, 1000, 1000, 1000, 1000, 1000, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1500, 1500]

hv = []
gain = []

for i in range(len(df)):
    guess = []
    guess.append([5000, 700, 100])
    guess.append([350, d[i], 200])

    #バックグラウンドの初期値
    background = 0

    #初期値リストの結合
    guess_total = []
    for t in guess:
        guess_total.extend(t)
    guess_total.append(background)
    coo = Path(df[i]).stem
    Coo = coo.split('V')
    f = open(df[i], 'r')
    data = f.read()
    data = data.split()
    data = [int(k) for k in data]
    data = [x for x in data if x != -1]
    data = [x for x in data if x != 0]
    data = [x for x in data if x != 4095]
    x = []
    y = []
    for k in range(bin):
        vq = [x for x in data if int(k*int_x) < x <= int((k+1)*int_x)]
        y.append(len(vq))
        x.append((k*int_x+(k+1)*int_x)/2)
    popt, pcov = curve_fit(func, x, y, p0=guess_total)
    
    volt = float(Coo[1]) * 12
    hv.append(volt)
    gain.append((popt[4]-popt[1])*0.25*10**(-12)/(100*1.6*10**(-19)))
    
    plt.figure()
    y_list = fit_plot(x, *popt)
    baseline = np.zeros_like(x) + popt[-1]
    for n,i in enumerate(y_list):
        plt.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
    plt.scatter(x, y, c='r')
    # plt.hist(data, bins=bin, range=(0, 4000))
    plt.xlim(0, 4000)
    plt.savefig(graph_dir+file_name+'_'+coo+'.png')
    plt.show()
    plt.close()
print(gain)
log_gain = [np.log10(i) for i in gain]
linear = np.polyfit(hv, log_gain, 1)
print(linear)
y_linear = [linear[0] * x_linear + linear[1] for x_linear in hv]
y_linear = [10**(i) for i in y_linear]
data = 10**(linear[0]*1060 + linear[1])
plt.figure()
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.plot(hv, y_linear, color='red')
plt.scatter(hv, gain)
plt.title('Gain vs HV_HV=1060, Gain={:.2e}'.format(data))
plt.yscale('log')
# plt.ylim(1*10**(6), 5*10**(7))
plt.xlabel('HV (V)')
plt.ylabel('Gain')
plt.grid()
#plt.legend()
plt.savefig(graph_dir+file_name+'gain_lin.png')

    

    