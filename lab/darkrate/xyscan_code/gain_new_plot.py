import pandas as pd
import numpy as np
from pathlib import Path
import csv
from scipy.optimize import curve_fit
import sys
from tqdm import tqdm
import glob
from mymath import func, fit_plot
import os
import matplotlib.pyplot as plt
from pylab import *



def gaussian_func(x, A, mu, sigma):
    return A * np.exp( - (x - mu)**2 / (2 * sigma**2))

dir_name = '2021-07-06-2028_45'

data_dir = './gaindata/' + dir_name + '/'
graph_dir ='./gain_graph/' + dir_name + '/'
# os.mkdir(graph_dir)
# os.mkdir('./gaindata/20210706_gain/')

bin = 100
max_range = 4100
min_range = 0
step = (max_range-min_range)/bin

dir = glob.glob(data_dir+'*')

p_ini = [[50, 1700, 400], [2000, 600, 50], [1200, 600, 100], [1000, 650, 200], [900, 700, 200], [500, 750, 200], [250, 750, 200], [200, 800, 250], [200, 850, 300], [300, 900, 300], [300, 1000, 300], [300, 1100, 350], [200, 1200, 400], [150, 1300, 400], [50, 1400, 400], [50, 1500, 400]]


for i in tqdm(dir):

    df = glob.glob(i+'/*.txt')

    coo = i.split('\\')
    coo = coo[1].split('_')
    coo_x = coo[0].split('-')[1]
    coo_y = coo[1].split('-')[1]

    with open('./gaindata/20210706_gain/x_{0}-y_{1}.csv'.format(coo_x, coo_y), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hv', 'gain'])
        f.close()

    data = []


    # os.mkdir(graph_dir+'x_{0}-y_{1}/'.format(coo_x, coo_y))
    num = 0

    for k in df:

        coo = k.split('V')
        coo_v = coo[1].split('.txt')[0]


        f = open(k, 'r')
        datas = f.read()
        datas = datas.split()
        datas = [float(x) for x in datas]
        datas = [x for x in datas if x != -1]
        datas = [x for x in datas if x != 0]
        datas = [x for x in datas if x != 4095]


        x = []
        y = []

        for s in range(bin):
            vq = [x for x in datas if min_range+step*s <= x < min_range+step*s+step]
            y.append(len(vq))
            x.append((min_range+step*s+min_range+step*s+step)/2)


        plt.figure()
        plt.scatter(x, y, color='black', s=5)


        popt_0, pcov_0 = curve_fit(gaussian_func, x, y, p0=[8000, 500, 80])
        xd = np.arange(min(x), max(x), 0.01)
        estimated_curve0 = gaussian_func(xd, popt_0[0], popt_0[1], popt_0[2])

        y_max = y.index(max(y))

        if num <= 6:

            del x[y_max-1:y_max+1]
            del y[y_max-1:y_max+1]

        else:

            del x[y_max-1:y_max+4]
            del y[y_max-1:y_max+4]

        popt, pcov = curve_fit(gaussian_func, x, y, p0=p_ini[num])
        estimated_curve = gaussian_func(xd, popt[0], popt[1], popt[2])

        hv = float(coo_v) * 12
        gain = (popt[1]-popt_0[1])*0.25*10**(-12)/(100*1.6*10**(-19))

        data.append([hv, gain])
        
        plt.plot(xd, estimated_curve0, color="r", label='ped={}'.format(popt_0[1]), linewidth = 0.5)
        plt.plot(xd, estimated_curve, color="g", label='ped={}'.format(popt[1]), linewidth = 0.5)
        plt.ylim(0, 2500)
        plt.legend()
        plt.savefig(graph_dir+'x_{0}-y_{1}/V_{2}.png'.format(coo_x, coo_y, coo_v))
        plt.close()

        num += 1

    with open('./gaindata/20210706_gain/x_{0}-y_{1}.csv'.format(coo_x, coo_y), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

    ddf = pd.read_csv('./gaindata/20210706_gain/x_{0}-y_{1}.csv'.format(coo_x, coo_y))
    x = ddf['hv']
    y = ddf['gain']
    plt.figure()
    plt.scatter(x, y, s=5)
    plt.ylim(100000, 100000000)
    plt.yscale('log')
    plt.savefig(graph_dir+'x_{0}-y_{1}/hv_gain.png'.format(coo_x, coo_y))
    plt.close()



df_list = glob.glob('./gaindata/20210706_gain/*.csv')
plt.figure()
for i in df_list:

    coo = i.split('\\')
    coo_x = coo[1].split('-')[0].split('_')[1]
    coo_y = coo[1].split('-')[1].split('_')[1]
    df = pd.read_csv(i)

    x = df['hv']
    y = df['gain']

    plt.scatter(x, y)
    plt.plot(x, y, linewidth=0.4, label='x_{}-y_{}'.format(coo_x, coo_y))

plt.show()





# data_dir = './gaindata/'
# file_dir = sys.argv[1]
# x_p = sys.argv[2]
# y_p = sys.argv[3]
# bin = 50
# max_range = 4100
# min_range = 0
# step = (max_range-min_range)/bin
# graph_dir = './gain_graph/'+file_dir+'_x-'+x_p+'_y-'+y_p

# # os.mkdir(graph_dir)


# df = glob.glob(data_dir+file_dir+'*/x-'+str(x_p)+'_y-'+str(y_p)+'/G*.txt')
# p_ini = [[10, 2000, 1000], [300, 900, 100], [10, 1100, 100], [10, 1200, 100], [10, 1300, 100], [10, 1400, 200], [15, 1500, 250], [10, 1500, 300], [50, 1200, 400], [30, 1300, 500], [20, 1300, 600], [15, 1400, 700], [15, 1400, 1000], [15, 1600, 1000], [15, 1700, 1000], [15, 1800, 1000], [15, 2100, 1000]]

# with open(graph_dir+'/hv_vs_gain.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['hv', 'gain'])
#     f.close()

# data = []

# for i in range(len(df)):

#     coo = Path(df[i]).stem
#     coo = coo.split('V')
#     volt = coo[1]

#     f = open(df[i], 'r')
#     datas = f.read()
#     datas = datas.split()
#     datas = [float(x) for x in datas]
#     datas = [x for x in datas if x != -1]
#     datas = [x for x in datas if x != 0]
#     datas = [x for x in datas if x != 4095]

#     x = []
#     y = []

#     for s in range(bin):
#         vq = [x for x in datas if min_range+step*s <= x < min_range+step*s+step]
#         y.append(len(vq))
#         x.append((min_range+step*s+min_range+step*s+step)/2)

#     xd = np.arange(min(x), max(x), 0.01)

#     popt_0, pcov_0 = curve_fit(gaussian_func, x, y, p0=[9000, 750, 50])
#     plt.figure()
#     plt.scatter(x, y, color='blue')
#     plt.ylim(0, 100)


#     y_max = y.index(max(y))
#     if i == len(df)-1 or i == len(df)-3 or i == len(df)-4 or i == 4 or i == 5 or i == 7:

#         del x[0:y_max+6]
#         del y[0:y_max+6]
        

#     elif i == 1 or i ==2 or i == 3:
#         del x[0:y_max+3]
#         del y[0:y_max+3]

#     elif i == 0 or i == 6:
#         del x[0:y_max+7]
#         del y[0:y_max+7]

#     elif i == len(df)-2:
#         del x[0:y_max+6]
#         del y[0:y_max+6]


#     else:
#         del x[0:y_max+2]
#         del y[0:y_max+2]


    

#     popt, pcov = curve_fit(gaussian_func, x, y, p0=p_ini[i])
#     print(popt_0[1], popt[1])

#     data.append([float(volt)*12, (popt[1]-popt_0[1])*0.25*10**(-12)/(100*1.6*10**(-19))])
    
#     estimated_curve0 = gaussian_func(xd, popt_0[0], popt_0[1], popt_0[2])
#     estimated_curve = gaussian_func(xd, popt[0], popt[1], popt[2])
#     plt.plot(xd, estimated_curve0, color="g")
#     plt.plot(xd, estimated_curve, color="r")
#     plt.savefig(graph_dir+'/GainMes_V-{0}.png'.format(volt))
#     plt.close()


# with open(graph_dir+'/hv_vs_gain.csv', 'a', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(data)
#     f.close()


# df = pd.read_csv(graph_dir+'/hv_vs_gain.csv')
# x = df['hv']
# y = df['gain']
# plt.figure()
# plt.scatter(x, y)
# plt.yscale('log')
# plt.show()



    

    
    













