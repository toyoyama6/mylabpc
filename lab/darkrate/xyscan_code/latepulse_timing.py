import matplotlib.pyplot as plt
import os
import sys
import glob
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np

data_dir = './dark_data/'
threshold = float(33)
file_name = input('enter file name: ') 
temp = input('enter temperature: ')
step_t = 0.2 # ns
min_R = 2700
max_R = 3200


# os.mkdir(data_dir+'ampli_data/')

length = len(open(data_dir+file_name+'.txt').readlines())
print(length)

data = []

for i in tqdm(range(length-1)):
    
    f = open(data_dir+file_name+'.txt')
    datas = f.readlines()[i+1].split()
    tri_datas = [-float(x) for x in datas[min_R:max_R]]
    max_data = max(tri_datas)
    k = tri_datas.index(max_data)
    t_tri = (len(tri_datas)-k)*step_t
    data.append(t_tri-t_tri)


    ave_datas = [-float(x) for x in datas[max_R+1:]]

    nlate = 0

    for k in range(len(ave_datas)-1):

        if ave_datas[k] <= threshold and ave_datas[k+1] > threshold:
            nlate += 1
            t_l = k*step_t
            data.append(t_l+t_tri)
        
        else:
            nlate += 0
    
    data.append([max_data, nlate])
    f.close()

with open(data_dir+'ampli_data/'+file_name+temp+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t_pulse'])
    f.close()

with open(data_dir+'ampli_data/'+file_name+temp+'.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
    f.close()

df = pd.read_csv(data_dir+'ampli_data/'+file_name+temp+'.csv')

x = df['max']
y = df['nlate']

plt.figure()
plt.hist(x, bins=25, range=(0, 200))
plt.show()

plt.figure()
plt.hist(y)
plt.show()


