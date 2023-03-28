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
from tqdm import tqdm

data_dir = './gaindata/'+sys.argv[1]
graph_dir = './gain_graph/'+sys.argv[1]
bin = 100
int_x = int(4000/bin)



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



d = [2000, 700, 750, 800, 900, 1000, 1000, 1000, 1000, 1000, 1000, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1500, 1500]
m = [20, 1000, 800, 5000, 200, 200, 100, 50, 50, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
w = [1000, 50, 50, 50, 50, 50, 100, 100, 100, 300, 300, 300, 500, 500, 500, 500, 500, 700, 700, 800, 1000]

dir = glob.glob(data_dir+'*/G*')

HV = []
GAIN = []

os.mkdir(graph_dir)
for i in tqdm(range(len(dir))):
    Coo = dir[i].split('_')
    r = np.sqrt((float(Coo[2])-93)**2+(float(Coo[3])-85)**2)
    if r <= 50:
        df = glob.glob(dir[i]+'/G*.txt')
        hv = []
        gain = []
        for k in range(len(df)):

            guess = []
            guess.append([5000, 600, 100])
            guess.append([m[k], d[k], w[k]])

            #バックグラウンドの初期値
            background = 0

            #初期値リストの結合
            guess_total = []
            for t in guess:
                guess_total.extend(t)
            guess_total.append(background)

            coo = Path(df[k]).stem
            coo = coo.split('_')
            coo = coo[1].split('V')
            f = open(df[k], 'r')
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
                y.append(len(vq)+1)
                x.append((k*int_x+(k+1)*int_x)/2)
            popt, pcov = curve_fit(func, x, y, p0=guess_total)

            hv.append(float(coo[1]) * 12)
            gain.append((popt[4]-popt[1])*0.25*10**(-12)/(100*1.6*10**(-19)))

            plt.figure()
            y_list = fit_plot(x, *popt)
            baseline = np.zeros_like(x) + popt[-1]
            for n,i in enumerate(y_list):
                plt.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
            plt.scatter(x, y, c='r')
            # plt.hist(data, bins=bin, range=(0, 4000))
            plt.ylim(0, 100)
            plt.xlim(0, 4095)
            plt.savefig(graph_dir+'/{0}_x-{1}_y-{2}.png'.format(coo[1], Coo[2], Coo[3]))
            plt.close()

        HV.append(hv)
        GAIN.append(gain)



    
    elif r > 50:
        print('BYE')
        print(dir[i])

print(HV, GAIN)

