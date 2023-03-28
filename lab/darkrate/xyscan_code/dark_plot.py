import pandas as pd
import sys
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from mymath import func, fit_plot
from scipy.optimize import curve_fit
import numpy as np
from pylab import *


data_dir = './dark_data/'
graph_dir = './dark_graph/'
bin = 40
min_range = 30
max_range = 190
step = (max_range-min_range)/bin

# csv_file = sys.argv[1]
txt_file = sys.argv[2]

# dfcsv1 = glob.glob(data_dir+csv_file+'*.csv')
# dfcsv2 = glob.glob(data_dir+csv_file+'*1.csv')

dftxt1 = glob.glob(data_dir+txt_file+'*ch1.txt')
dftxt2 = glob.glob(data_dir+txt_file+'*ch2.txt')

# csv1 = pd.read_csv(dfcsv1[0])
# csv2 = pd.read_csv(dfcsv2[0])

# y1 = csv1['rate_ave']
# y2 = csv2['rate_ave']
# x1 = csv1['thre']
# x2 = csv2['thre']
# yerr1 = csv1['rate_std']
# yerr2 = csv2['rate_std']


# plt.figure()
# plt.title('Threshold vs rate')
# plt.rcParams['font.size']=15
# plt.subplots_adjust(bottom=0.2, left=0.2)
# plt.errorbar(x1, y1, yerr=yerr1, fmt='o', color='blue', ms=3)
# plt.errorbar(x2, y2, yerr=yerr2, fmt='o', color='red')
# plt.yscale('log')
# plt.xlabel('threshold (mV)')
# plt.ylabel('rate (Hz)')
# # plt.xlim(10, 150)
# plt.savefig(graph_dir+csv_file+'_darkrate.png')


length = len(open(dftxt1[0]).readlines())
data = []

for i in tqdm(range(length-1)):

    f = open(dftxt1[0], 'r')
    datas = f.readlines()[i+1].split()
    datas = [float(x) for x in datas[2800:3200]]
    data.append(-min(datas))

x = []
y = []


for i in range(bin):
    vq = [x for x in data if min_range+step*i <= x < min_range+step*i+step]
    y.append(len(vq))
    x.append((min_range+step*i+min_range+step*i+step)/2)


guess = []
guess.append([60, 40, 10])
guess.append([25, 110, 30])

#バックグラウンドの初期値
background = 0

#初期値リストの結合
guess_total = []
for t in guess:
    guess_total.extend(t)
guess_total.append(background)

popt, pcov = curve_fit(func, x, y, p0=guess_total)

print(popt)

plt.figure()
y_list = fit_plot(x, *popt)
baseline = np.zeros_like(x) + popt[-1]
for n,i in enumerate(y_list):
    plt.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
plt.scatter(x, y, c='r')
plt.savefig(graph_dir+txt_file+'_gaus_darkamp.png')

plt.figure()
plt.title('amplitude')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.hist(data, bins=bin, range=(min_range, max_range))
plt.xlabel('amplitude (mV)')
plt.ylabel('count')
plt.savefig(graph_dir+txt_file+'_darkamp.png')
