import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


df1_1 = pd.read_csv('./gaindata/20210706_gain/x_93-y_85.csv')

x1_1 = df1_1['hv']
y1_1 = df1_1['gain']
del x1_1[1]
del y1_1[1]



df2_1 = pd.read_csv('./gaindata/20210706_gain/x_93-y_63.csv')
df2_2 = pd.read_csv('./gaindata/20210706_gain/x_93-y_107.csv')
df2_3 = pd.read_csv('./gaindata/20210706_gain/x_71-y_85.csv')
df2_4 = pd.read_csv('./gaindata/20210706_gain/x_115-y_85.csv')

x2_1 = df2_1['hv']
y2_1 = df2_1['gain']
del x2_1[1]
del y2_1[1]

x2_2 = df2_2['hv']
y2_2 = df2_2['gain']
del x2_2[1]
del y2_2[1]

x2_3 = df2_3['hv']
y2_3 = df2_3['gain']
del x2_3[1]
del y2_3[1]

x2_4 = df2_4['hv']
y2_4 = df2_4['gain']
del x2_4[1]
del y2_4[1]



df3_1 = pd.read_csv('./gaindata/20210706_gain/x_50-y_85.csv')
df3_2 = pd.read_csv('./gaindata/20210706_gain/x_93-y_42.csv')
df3_3 = pd.read_csv('./gaindata/20210706_gain/x_93-y_128.csv')
df3_4 = pd.read_csv('./gaindata/20210706_gain/x_136-y_85.csv')

x3_1 = df3_1['hv']
y3_1 = df3_1['gain']
del x3_1[1]
del y3_1[1]

x3_2 = df3_2['hv']
y3_2 = df3_2['gain']
del x3_2[1]
del y3_2[1]

x3_3 = df3_3['hv']
y3_3 = df3_3['gain']
del x3_3[1]
del y3_3[1]

x3_4 = df3_4['hv']
y3_4 = df3_4['gain']
del x3_4[1]
del y3_4[1]




plt.figure()
plt.scatter(x1_1, y1_1, s=30, label='center', color='black')

plt.scatter(x2_1, y2_1, s=15, label='u', color='red', marker='^')
plt.scatter(x2_2, y2_2, s=15, label='b', color='red', marker='v')
plt.scatter(x2_3, y2_3, s=15, label='l', color='red', marker='<')
plt.scatter(x2_4, y2_4, s=15, label='r', color='red', marker='>')

plt.scatter(x3_1, y3_1, s=15, label='ll', color='blue', marker='<')
plt.scatter(x3_2, y3_2, s=15, label='uu', color='blue', marker='^')
plt.scatter(x3_3, y3_3, s=15, label='bb', color='blue', marker='v')
plt.scatter(x3_4, y3_4, s=15, label='rr', color='blue', marker='>')
plt.xlim(850, 1250)
plt.ylim(700000, 30000000)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
