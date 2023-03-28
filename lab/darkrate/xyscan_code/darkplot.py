import pandas as pd
import matplotlib.pyplot as plt

data_dir = './dark_data/'
graph_dir = './dark_graph/'

df1 = pd.read_csv(data_dir+'test.csv')
df2 = pd.read_csv(data_dir+'20210701-1951_discri.csv')
df3 = pd.read_csv(data_dir+'20210702-1404_discri.csv')

x1 = df1['thre']/10
y1 = df1['rate_ave']
yerr1 = df1['rate_std']

x2 = df2['thre']/10
y2 = df2['rate_ave']
yerr2 = df2['rate_std']

x3 = df3['thre']/10
y3 = df3['rate_ave']
yerr3 = df3['rate_std']

plt.figure()

plt.title('Threshold vs rate')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.errorbar(x1, y1, yerr=yerr1, fmt='o', color='blue', ms=3, label='room(20'+chr(176)+'C)')
plt.errorbar(x2, y2, yerr=yerr2, fmt='o', color='red', ms=3, label='freez(-9'+chr(176)+'C)')
plt.errorbar(x3, y3, yerr=yerr3, fmt='o', color='green', ms=3, label='freez(-36.6'+chr(176)+'C)')

plt.yscale('log')
plt.xlabel('threshold (mV)')
plt.ylabel('rate (Hz)')
plt.legend()
# plt.xlim(10, 150)
plt.savefig(graph_dir+'discri_darkrate.png')