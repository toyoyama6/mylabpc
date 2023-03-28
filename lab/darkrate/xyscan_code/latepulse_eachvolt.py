import matplotlib.pyplot as plt
import os
import sys
import glob
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np

data_dir = './dark_data/'
threshold = [x for x in range(50, 200, 5)]
print(threshold)
file_name = ['20210706-1531_thre-32_ch1'] #room,-9, -36, -44
csv_file = ['-44.5_eachV.csv']
# os.mkdir(data_dir+'ampli_data/')

for s in range(len(file_name)):

    length = len(open(data_dir+file_name[s]+'.txt').readlines())
    print(length)

    data = []

    for t in tqdm(threshold):

        ndata = []

        for i in range(length-1):

            if s == 0 and s == 1:
            
                f = open(data_dir+file_name[s]+'.txt')
                datas = f.readlines()[i+1].split()
                tri_datas = [-float(x) for x in datas[2800:3150]] #
                ave_datas = [-float(x) for x in datas[3151:]]

            else:
            
                f = open(data_dir+file_name[s]+'.txt')
                datas = f.readlines()[i+1].split()
                tri_datas = [-float(x) for x in datas[2450:2800]] #
                ave_datas = [-float(x) for x in datas[2801:]]

            nlate = 0
            

            for k in range(len(ave_datas)-1):

                if ave_datas[k] <= t and ave_datas[k+1] > t:
                    nlate += 1
                
                else:
                    nlate += 0

            ndata.append(nlate)

            
            
        data.append([t, np.mean(ndata)])
        f.close()

    with open(data_dir+'ampli_data/'+csv_file[s], 'w', newline='') as f:
    # with open(data_dir+'ampli_data/'+'-44.5.csv', 'w', newline='') as f:
    # with open(data_dir+'ampli_data/'+'-9.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['thre', 'nlate'])
        f.close()

    with open(data_dir+'ampli_data/'+csv_file[s], 'a', newline='') as f:
    # with open(data_dir+'ampli_data/'+'-44.5.csv', 'a', newline='') as f:
    # with open(data_dir+'ampli_data/'+'-9.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()