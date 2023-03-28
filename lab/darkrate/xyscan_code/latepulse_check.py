import matplotlib.pyplot as plt
import os
import sys
import glob
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np

data_dir = './dark_data/'
threshold = float(50)
file_name = input('enter file name: ') 
temp = input('enter temperature: ')
#room
# file_name = '20210706-1531_thre-32_ch1' #-44.5
# file_name = '20210702-1437__-36_ch1' #-36
# file_name ='20210701-2032_ch2'

# os.mkdir(data_dir+'ampli_data/')

length = len(open(data_dir+file_name+'.txt').readlines())
print(length)

data = []

for i in tqdm(range(length-1)):
    
    f = open(data_dir+file_name+'.txt')
    datas = f.readlines()[i+1].split()
    tri_datas = [-float(x) for x in datas[1300:1600]]
    ave_datas = [-float(x) for x in datas[1600:]]

    nlate = 0

    for k in range(len(ave_datas)-1):

        if ave_datas[k] <= threshold and ave_datas[k+1] > threshold:
            nlate += 1
        
        else:
            nlate += 0
    
    data.append([max(tri_datas), nlate])
    f.close()

with open(data_dir+'ampli_data/'+file_name+temp+'.csv', 'w', newline='') as f:
# with open(data_dir+'ampli_data/'+'-44.5.csv', 'w', newline='') as f:
# with open(data_dir+'ampli_data/'+'-9.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['max', 'nlate'])
    f.close()

with open(data_dir+'ampli_data/'+file_name+temp+'.csv', 'a', newline='') as f:
# with open(data_dir+'ampli_data/'+'-44.5.csv', 'a', newline='') as f:
# with open(data_dir+'ampli_data/'+'-9.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()

df = pd.read_csv(data_dir+'ampli_data/'+file_name+temp+'.csv')
# df = pd.read_csv(data_dir+'ampli_data/'+'-9.csv')
x = df['max']
y = df['nlate']

plt.figure()
plt.hist(x, bins=25, range=(0, 200))
plt.show()

plt.figure()
plt.hist(y, bins=15, range=(-0.5, 14.5))
plt.title('Number of after-pulse T=-10')
plt.xlabel('N of after-pulse')
plt.yscale('log')
plt.ylim(0, 1000)
plt.savefig('./dark_graph/N_of_afterpulse-hist-T_{}.png'.format(file_name))
plt.show()


