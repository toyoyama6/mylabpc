import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import integrate
import sys

csv_data_dir  = './csv_data/'
wogel_list = ['2022-01-31-1215_58_bared_PMT', '2022-02-02-1647_48_bared_PMT', '2022-01-30-1003_58_bared_PMT',  '2022-02-03-1308_15_bared_PMT', '2022-01-30-1954_17_bared_PMT']
wgel_list = ['2022-02-07-1928_02_gel_PMT', '2022-02-08-1556_40_gel_PMT', '2022-02-09-1724_26_gel_PMT', '2022-02-10-1741_44_gel_PMT', '2022-02-11-1654_21_gel_PMT']
degree_list = [0, 10, 30, 40, 45]

Q_diff_list = []
for i in range(5): 
    df_scan1 = pd.read_csv(csv_data_dir + wogel_list[i] + '.csv')
    df_scan2 = pd.read_csv(csv_data_dir + wgel_list[i] + '.csv')
    # df_scan['charge'] / np.cos(np.radians(degree_list[i]))
    # diff_list = []
    for j in range(len(df_scan1)):
        diff = df_scan2["charge"] - df_scan1["charge"]
    # print(diff)
    df_scan1.insert(3, "diff", diff, True)
    # print(df_scan1)
    x = df_scan1["x"].unique() 
    y = df_scan1["y"].unique() 
    
    value_list = []
    for k in y:
        tf = df_scan1[df_scan1['y'] == k]
        # print(tf)
        tf = tf.sort_values('x')
        val = list(tf["diff"])
        val.append(0)
        val.insert(0, 0)
        value_list.append(val)
    
    # print(value_list)
    zero = []
    for j in range(len(value_list[0])):
        zero.append(0)

    value_list.append(zero)
    value_list.insert(0, zero)

    
    x = x - 145
    y = y -75
    x = np.append(-100, x)
    x = np.append(x, 100)
    y = np.append(-100, y)
    y = np.append(y, 100)

    X, Y = np.meshgrid(x, y)
    # print(value_list)
    plt.figure()
    plt.pcolormesh(X, Y, value_list)
    plt.ylim(-100, 100)
    plt.xlim(-100, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'{degree_list[i]}' + 'degree', fontsize = 18)
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('y', fontsize = 16)
    plt.colorbar().set_label('charge[pc]', fontsize = 16)
    # plt.savefig('./graph/' + f'{degree_list[i]}' + 'degree' +'_increasing_map.png', bbox_inches = 'tight')
    # plt.show() 
    plt.close()
    
    Q_diff_list.append(df_scan1["diff"].sum())
    
print(Q_diff_list)

  
    