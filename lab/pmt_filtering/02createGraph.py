# ある検出器、ある距離について6つのフィルターのデータから1つのグラフを描画する

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
from scipy.stats.stats import CumfreqResult
matplotlib.use("tkagg")
matplotlib.rc("font", family="Noto Sans CJK JP")
import pandas as pd
import os
from scipy.stats import norm
from tqdm import tqdm

detector = "pmt"
distance = "0"
channel = "0"

numOfLines = 100

def createDataPoint(filename):
    readFile = filename

    maxes = []*numOfLines

    f = open(readFile, "r")

    for i in range(numOfLines + 2):
        line = f.readline()
        if i == 0:
            continue
        elif line:
            values = np.array(line.split())
            values = np.array(values, dtype="float16")
            # mVからVに変換
            values = 1000*values
            """
            !!!!!!!!!!!!!!!
            mppcならmax
            pmtならminの正負逆転
            !!!!!!!!!!!!!!!
            """
            maxes.append(-min(values))
            #plt.plot(values)
        else:
            break

    # 確認用
    #plt.show()
    f.close()

    #print(maxes)
    #print(len(maxes))

    params = norm.fit(maxes)
    # 確認用
    """
    print(params)
    mu = params[0]
    sigma = params[1]
    print(mu - sigma, mu + sigma)
    """
    return params

#for filter in tqdm([5, 10, 25, 32, 50, 100]):
for filter in tqdm([5, 10, 25, 32, 50, 100]):
    filename = detector + "_" + distance + "_" + str(filter).zfill(3) + "_" + "ch" + str(channel) + ".txt"
    mu, sigma = createDataPoint(filename)
    # 確認用
    print(filename)
    # エラーバー付きでプロットする
    # +-1σ = 68%, +-2σ = 95%
    print(filter, mu, sigma)
    plt.errorbar(filter, mu, yerr=2*sigma, fmt='o', capsize=5, markersize=10, ecolor='black', markeredgecolor = "black", color='w')

plt.title(detector + distance)
plt.xlabel("filter [%]")
plt.ylabel("voltage [V]")
plt.show()