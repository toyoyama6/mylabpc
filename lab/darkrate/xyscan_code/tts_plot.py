import pandas as pd 
import glob
from pathlib import Path
import csv
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

def getNearestValue(list, num):
    
    idx = np.abs(np.asarray(list) - num).argmin()
    return idx

csv_data_dir = './csv_TTS/'
data_dir = './TTSdata/'
graph_dir = './TTSgraph/'
thre = 0.3

data_d = sys.argv[1]

ch0_df = glob.glob(data_dir+data_d+'*/'+'*_ch0.txt')
ch1_df = glob.glob(data_dir+data_d+'*/'+'*_ch1.txt')

f = open(ch0_df[0], 'r')
t_data = f.readlines()[0]
t_data = t_data.split()
t_data = [float(x) for x in t_data]
f.close()
t = t_data[1] - t_data[0]
step = t_data[2]-1
t_step = t/step

# Tri = []

# for i in tqdm(range(len(ch0_df))):

#     length = len(open(ch0_df[i]).readlines())

#     for k in range(length-1):

#         f = open(ch0_df[i], 'r')
#         tri = f.readlines()[k+1].split()
#         # tri = tri.split()
#         tri = tri[1100:1400]
#         tri = [float(x) for x in tri]
#         f.close()
#         value = getNearestValue(tri, (max(tri)-min(tri))*thre) + 1100 + 1
#         Tri.append(value)

# params = norm.fit(Tri)
# C = params[0]
# print(params)

# # C = 1273.7 #thre=0.5
C = 1269.4 #thre=0.3



# with open(csv_data_dir+data_d+'_timing_'+str(thre)+'_.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['X', 'Y', 'Value'])
#     f.close()



# for i in tqdm(range(len(ch1_df))):

#     wid = []

#     length = len(open(ch1_df[i]).readlines())

#     coo = Path(ch1_df[i]).stem
#     Coo = coo.split('_')

#     for k in range(length-1):
#         f = open(ch1_df[i], 'r')
#         sig = f.readlines()[k+1].split()
#         sig = [float(x) for x in sig[1400:1650]]
#         value = getNearestValue(sig, (max(sig)-min(sig))*(1-thre)) + 1400 + 1
#         if min(sig) > -3:
#             wid.append(0)
#         elif min(sig) <= -3:
#             wid.append((value-C)*t_step*10**9)

#     params = norm.fit(wid)
#     datas = [Coo[2], Coo[3], params[0]]

#     # if 0 in wid:
#     #     datas = [Coo[2], Coo[3], None]
#     # else:
#     #     params = norm.fit(wid)
#     #     datas = [Coo[2], Coo[3], params[0]]

#     with open(csv_data_dir+data_d+'_timing_'+str(thre)+'_.csv', 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(datas)
#         f.close()

df = pd.read_csv(csv_data_dir+data_d+'_timing_'+str(thre)+'_.csv')
df_pivot = pd.pivot_table(data=df, values='Value', index='Y', columns='X', aggfunc=np.mean)
plt.figure()
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.title('Timing XYscan map')
sns.heatmap(df_pivot, square=True, vmin=100, vmax=125)
plt.savefig(graph_dir+data_d+'_Tinming_'+str(thre)+'_xyscan_test.png')










# for i in tqdm(range(len(ch0_df))):

#     tri_coo = Path(ch0_df[i]).stem
#     tri_Coo = tri_coo.split('_')

#     sig_coo = Path(ch1_df[i]).stem
#     sig_Coo = sig_coo.split('_')

#     if sig_Coo[2] == tri_Coo[2] and sig_Coo[3] == tri_Coo[3]:

#         t_wid = []
#         length = len(open(ch0_df[i]).readlines())

#         for k in range(length-1):

            
#             f = open(ch0_df[i], 'r')
#             tri = f.readlines()[k+1]
#             tri = tri.split()
#             tri = [float(x) for x in tri]
#             f.close()
#             max_tri = max(tri)
#             tri_thre_value = thre*max_tri
#             tri_thre_data = [x for x in tri if x >= tri_thre_value]
#             # t_tri = (len(tri) - len(tri_thre_data)) * t_step


#             f = open(ch1_df[i], 'r')
#             sig = f.readlines()[k+1]
#             sig = sig.split()
#             sig = [float(x) for x in sig]
#             f.close()
#             min_sig = min(sig)
#             sig_thre_value = thre*min_sig
#             sig_thre_data = [x for x in sig if x <= sig_thre_value]

#             if min_sig >= -4:

#                 width = 0
            
#             elif min_sig < -4:
                
#                 # t_sig = (len(sig) - sig.index(sig_thre_data[0])) * t_step
#                 width = ((len(sig) - sig.index(sig_thre_data[0]))-(len(tri) - len(tri_thre_data)))*t_step

#             t_wid.append(width)

#         params = norm.fit(t_wid)
#         if params[0] <= 2*10**(-8):
#             value = 0
#         else:
#             value = params[0]
#         datas = [tri_Coo[2], tri_Coo[3], value]
#         with open(csv_data_dir+data_d+'_TTS.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(datas)
#             f.close()

#     else:
#         sys.exit()








