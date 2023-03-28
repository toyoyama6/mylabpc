import pandas as pd
import serial
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

data_dir = './dark_data/'
graph_dir = './dark_graph/'


df = pd.read_csv(data_dir+'20210708-2037.csv')
df2 = pd.read_csv(data_dir+'20210701-1951_discri.csv')

x = df['thre']
y = df['rate_ave']
y_err = df['rate_std']

x2 = df2['thre']
y2 = df2['rate_ave']
y_err2 = df2['rate_std']

plt.figure()
plt.title('DarkRate new&old T=-10')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.errorbar(x, y, yerr=y_err, fmt='o', label='new'+chr(176)+'C', ms=3, c='b')
# plt.errorbar(x1, y1, yerr=y_err1, fmt='o', label='23', ms=3, c='k')
plt.errorbar(x2, y2, yerr=y_err2, fmt='o', label='old'+chr(176)+'C', ms=3, c='r')
# plt.errorbar(x3, y3, yerr=y_err3, fmt='o', label='-44.5'+chr(176)+'C', ms=3, c='g')
plt.xlabel('Threshold (mV)')
plt.ylabel('Darkrate (Hz)')
plt.yscale('log')
plt.legend()
plt.savefig(graph_dir+'discri_new_old.png')
plt.show()

####################################################

df = pd.read_csv(data_dir+'20210708-2026.csv')
df1 = pd.read_csv(data_dir+'20210701-1718.csv')

x = df['thre']
y = df['rate_ave']
y_err = df['rate_std']

x1 = df1['thre']
y1 = df1['rate_ave']
y_err1 = df1['rate_std']

plt.figure()
plt.title('DarkRate new&old T=-10')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.errorbar(x, y, yerr=y_err, fmt='o', label='new'+chr(176)+'C', ms=3, c='b')
plt.errorbar(x1, y1, yerr=y_err1, fmt='o', label='old'+chr(176)+'C', ms=3, c='r')
# plt.errorbar(x2, y2, yerr=y_err2, fmt='o', label='-44.5'+chr(176)+'C', ms=3, c='k')
# plt.errorbar(x3, y3, yerr=y_err3, fmt='o', label='23'+chr(176)+'C', ms=3, c='g')
plt.xlabel('Threshold (mV)')
plt.ylabel('Darkrate (Hz)')
plt.yscale('log')
plt.legend()
# plt.savefig('.png')
plt.savefig(graph_dir+'circuit_new_old.png')
plt.show()





######measured by hand


df = pd.read_csv(data_dir+'20210702-1404_discri.csv')
df1 = pd.read_csv(data_dir+'test.csv')
df2 = pd.read_csv(data_dir+'20210701-1951_discri.csv')
df3 = pd.read_csv(data_dir+'20210706-1459_discri.csv')
x = df['thre']
y = df['rate_ave']
y_err = df['rate_std']

x1 = df1['thre']
y1 = df1['rate_ave']
y_err1 = df1['rate_std']

x2 = df2['thre']
y2 = df2['rate_ave']
y_err2 = df2['rate_std']

x3 = df3['thre']
y3 = df3['rate_ave']
y_err3 = df3['rate_std']

plt.figure()
plt.title('DarkRate(1)')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)

plt.errorbar(x1, y1, yerr=y_err1, fmt='o', label='23', ms=3, c='k')
plt.errorbar(x2, y2, yerr=y_err2, fmt='o', label='-9'+chr(176)+'C', ms=3, c='r')
plt.errorbar(x, y, yerr=y_err, fmt='o', label='-36.6'+chr(176)+'C', ms=3, c='b')
plt.errorbar(x3, y3, yerr=y_err3, fmt='o', label='-44.5'+chr(176)+'C', ms=3, c='g')
plt.xlabel('Threshold (mV)')
plt.ylabel('Darkrate (Hz)')
plt.yscale('log')
plt.legend()
plt.savefig(graph_dir+'discri_eachT.png')
plt.show()


######measured by auto

df = pd.read_csv(data_dir+'20210702-1002.csv')
df1 = pd.read_csv(data_dir+'20210701-1718.csv')
df2 = pd.read_csv(data_dir+'20210706-1347.csv')
df3 = pd.read_csv(data_dir+'20210706-1913.csv')
x = df['thre']
y = df['rate_ave']
y_err = df['rate_std']

x1 = df1['thre']
y1 = df1['rate_ave']
y_err1 = df1['rate_std']

x2 = df2['thre']
y2 = df2['rate_ave']
y_err2 = df2['rate_std']

plt.figure()
plt.title('DarkRate(2)')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.errorbar(x3, y3, yerr=y_err3, fmt='o', label='23'+chr(176)+'C', ms=3, c='k')

plt.errorbar(x1, y1, yerr=y_err1, fmt='o', label='-9'+chr(176)+'C', ms=3, c='r')
plt.errorbar(x, y, yerr=y_err, fmt='o', label='-36.6'+chr(176)+'C', ms=3, c='b')
plt.errorbar(x2, y2, yerr=y_err2, fmt='o', label='-44.5'+chr(176)+'C', ms=3, c='g')

plt.xlabel('Threshold (mV)')
plt.ylabel('Darkrate (Hz)')
plt.yscale('log')
plt.legend()
# plt.savefig('.png')
plt.savefig(graph_dir+'circuit_eachT.png')
plt.show()



df1 = pd.read_csv(data_dir+'20210706-1913.csv')
df2 = pd.read_csv(data_dir+'test.csv')


x1 = df1['thre']
y1 = df1['rate_ave']
y_err1 = df1['rate_std']

x2 = df2['thre']
y2 = df2['rate_ave']
y_err2 = df2['rate_std']

plt.figure()
plt.title('DarkRate(3)')
plt.rcParams['font.size']=15
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.errorbar(x1, y1, yerr=y_err1, fmt='o', label='circuit', ms=3)
plt.errorbar(x2, y2, yerr=y_err2, fmt='o', label='discri', ms=3)
plt.xlabel('Threshold (mV)')
plt.ylabel('Darkrate (Hz)')
plt.yscale('log')
plt.legend()
plt.savefig(graph_dir+'discri_circuit.png')
# plt.savefig('.png')
plt.show()



