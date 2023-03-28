import pandas as pd
import numpy as np
from scipy import integrate

def nezi(list_name):
    for i, file in enumerate(list_name): 
        df_scan = pd.read_csv(csv_data_dir + file + '.csv')
        x = df_scan["x"].unique() 
        y = df_scan["y"].unique()
        rm_xlist = []
        rm_ylist = []
        for j in range(len(x)):
            for k in range(len(x)):
                R = np.sqrt((x[j] - degree_list[i][1]) ** 2 + (y[k] - 75) ** 2)
                if R > 65:
                    rm_xlist.append(x[j])
                    rm_ylist.append(y[k])
        for l in range(len(rm_xlist)):
            # if (df_scan["x"] == rm_xlist[l]) & (df_scan["y"] == rm_ylist[l]):
            df_scan["charge"][(df_scan["x"] == rm_xlist[l]) & (df_scan["y"] == rm_ylist[l])] = 0
            
            
def light_check():
    delta_t = (6.008 * 10 ** -7 - (-6.016 * 10 ** -7)) / 1504
    t =  np.arange(-6.016 * 10 ** -7, 6.008 * 10 ** -7, delta_t)

    readname_list = ['./data/PMT/2022-01-31-1215_58_bared_PMT/', './data/PMT/2022-02-02-1647_48_bared_PMT/', './data/PMT/2022-01-30-1003_58_bared_PMT/', './data/PMT/2022-02-03-1308_15_bared_PMT/', './data/PMT/2022-01-30-1954_17_bared_PMT/', './data/PMT/2022-02-07-1928_02_gel_PMT/', './data/PMT/2022-02-08-1556_40_gel_PMT/', './data/PMT/2022-02-09-1724_26_gel_PMT/', './data/PMT/2022-02-10-1741_44_gel_PMT/', './data/PMT/2022-02-11-1654_21_gel_PMT/']      
    n_list = ['light_check_start_ch1.txt', 'light_check_end_ch1.txt']
    std_list = []
    for readname in readname_list:
        a_list = []
        for n in n_list:
            data = np.loadtxt(readname + n, skiprows=1)
            data = -data
            charge_list = []
            # plt.figure()
            for k in range(len(data)):
                data[k] = data[k] - np.mean(data[k][:50])
                # plt.figure()
                # plt.plot(data[k])        
                value = integrate.simps(data[k][800:1100], t[800:1100]) 
                charge = (value / 50) * 10 ** 12
                charge_list.append(charge) 
            # print(np.mean(charge_list),'pc')
            a_list.append(np.mean(charge_list))
        std_list.append(np.std(a_list))
    return std_list
        
        