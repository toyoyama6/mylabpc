import pandas as pd 
import glob
from pathlib import Path
import csv
import seaborn as sns
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mymath import Pulse


#data directry
# data_dir = './data/'
csv_data_dir = './csv_data/'
graph_dir = './graph/'


#set name
date = input('what date data do you want? (ex:2021-05-21_20)Type by hour!!!!\n: ')
if date == '2021-05-17' or date == '2021-05-28':
    dx = 92.1
    dy = 81.7
elif date == '2021-0609':
    dx = 83.8
    dy = 65
else :
    dx = 84
    dy = 75

detector = input('\nWhich detector did you use?\n1 : PMT\n2 : MPPC\n>>>>>>>')
if detector == str(1):
    detec = 'PMT'
    plt_name = 'XY scan'
    data_dir = './data/PMT/'
    wx = 50
    xc = 0
    yc = 0
    #size = 38.1 #3 inch PMT
    size = 51 #4 inch PMT
elif detector == str(2):
    detec = 'MPPC'
    plt_name = 'XY laser scan'
    data_dir = './data/MPPC/'
    dx = 85
    dy = 5
    wx = 8
    xc = 0
    yc = 0
    size = 3
else:
    print('Wrong!!')


df = glob.glob(data_dir + date + '*/' + '*.txt')
coo = Path(df[0]).stem
Coo = coo.split('_')
t = '_' + Coo[1] + '_'


#set option
option = input('\nyou choise option!!\n0 : No thank you\n1 : make relative plot\n2 : cutting plot\n3 : full option\n>>>>>>>')
if detector == str(1):
    if option == str(0) or option == str(2):
        max_value = 1
        dx = 0
        dy = 0
        opt = '_row-data_'

    elif option == str(1) or option == str(3):
        L = []
        for i in range(len(df)):
            f = open(df[i], 'r')
            value = f.readlines()[1]
            value = value.split()
            value = [float(k) for k in value]
            value = [x for x in value if x != -1]
            params = norm.fit(value)
            L.append(params[0])
        max_value = max(L)
        opt = '_relative-data_'

elif detector == str(2):
    if option == str(0) or option == str(2):
        max_value = 1
        dx = 0
        dy = 0
        opt = '_row-data_'

    elif option == str(1) or option == str(3):
        max_value = 1
        opt = '_row_data_'


#set csv file
file_name = date+t+opt+detec
if os.path.exists(csv_data_dir+file_name+'.csv') == bool(True):
    say = input('\nCan I remove this file('+ csv_data_dir+file_name+'.csv)\n0 : No\n1 : Yes\n>>>>>>>')
    if say == str(0):
        print('bye bye')
        sys.exit()
    elif say == str(1):
        print('remove file')
        os.remove(csv_data_dir+file_name+'.csv')
    else:
        print('Wrong!!')
        sys.exit()
elif os.path.exists(csv_data_dir+file_name+'.csv') == bool(False):
    print('go ahead')

with open(csv_data_dir+file_name+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['X', 'Y', 'Value'])
    f.close()


#make csv file
for i in range(len(df)):
    coo = Path(df[i]).stem
    Coo = coo.split('_')
    f = open(df[i], 'r')
    value = f.readlines()[1]
    value = value.split()
    value = [float(k) for k in value]
    value = [x for x in value if x != -1]
    params = norm.fit(value)
    f.close()
    with open(csv_data_dir+file_name+'.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        data = [round(float(Coo[2])-dx, 1), round(float(Coo[3])-dy, 1), params[0]/max_value]
        writer.writerow(data)
        f.close()


#plot colormap
# if option == str(1) or option == str(3):
#     df_scan = pd.read_csv(csv_data_dir+file_name+'.csv')
#     df_scan = df_scan[(df_scan['X'] >= -wx) & (df_scan['X'] <= wx) & (df_scan['Y'] >= -wx) & (df_scan['Y'] < wx)]
#     df_scan_pivot = pd.pivot_table(data=df_scan, values='Value', index='X', columns='Y', aggfunc=np.mean)
#     plt.figure()
#     plt.title(plt_name+' map')
#     sns.heatmap(df_scan_pivot, square=True, xticklabels=10, yticklabels=10)
#     plt.savefig(graph_dir+file_name+'_xyscan_square.png')

df_scan = pd.read_csv(csv_data_dir+file_name+'.csv')
df_scan_pivot = pd.pivot_table(data=df_scan, values='Value', index='Y', columns='X', aggfunc=np.mean)
plt.figure()
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.title(plt_name+' map')
sns.heatmap(df_scan_pivot, square=True)
plt.savefig(graph_dir+file_name+'_xyscan.png')


#plot cutting scatter
if option == str(2) or option == str(3):
    pulse = Pulse()
    df = pd.read_csv(csv_data_dir+file_name+'.csv')
    xdf = df[df.Y == yc]
    ydf = df[df.X == xc]
    xdf_x = xdf['X']
    ydf_y = ydf['Y']
    xdf_value = xdf['Value']
    ydf_value = ydf['Value']
    data1 = pulse.cal_time(ts=xdf_x, ys=xdf_value,upper_threshold=0.8, lower_threshold=0.1, ymax=1, ymin=0)
    data2 = pulse.cal_time(ts=ydf_y, ys=ydf_value,upper_threshold=0.8, lower_threshold=0.1, ymax=1, ymin=0)
    x_tw = data1['tw']
    y_tw = data2['tw']
    print(x_tw, y_tw)
    plt.figure()
    plt.rcParams['font.size']=15
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.title(plt_name+' map(x-axis)')
    plt.ylabel('relative intensity')
    plt.scatter(xdf_x, xdf_value)
    plt.axvline(x=size, color='r')
    plt.axvline(x=-size, color='r')
    plt.savefig(graph_dir+file_name+'_xyscan_x-axis.png')

    plt.figure()
    plt.rcParams['font.size']=15
    plt.subplots_adjust(bottom=0.2, left=0.2)   
    plt.title(plt_name+' map(y-axis)')
    plt.ylabel('relative intensity')
    plt.scatter(ydf_y, ydf_value)
    plt.axvline(x=size, color='r')
    plt.axvline(x=-size, color='r')
    plt.savefig(graph_dir+file_name+'_xyscan_y-axis.png')

    plt.figure()
    plt.rcParams['font.size']=15
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.title(plt_name+' map(xy-axis)')
    plt.ylabel('relative intensity')
    plt.scatter(xdf_x, xdf_value, label='x-axis_'+str(x_tw))
    plt.scatter(ydf_y, ydf_value, label='y-axis_'+str(y_tw))
    plt.axvline(x=size, color='r')
    plt.axvline(x=-size, color='r')
    plt.legend()
    plt.savefig(graph_dir+file_name+'_xyscan_xy-axis.png')