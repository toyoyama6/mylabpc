import csv 
import matplotlib.pyplot as plt
import numpy as np


with open ('./data/darkrate_count.csv', 'r', encoding='utf-8-sig') as f: 
    reader = csv.reader(f)
    Hz_list = []
    mv_list = []
    var_list = []

    for line in reader:
        line = [int(float(i)) for i in line]
        var = np.std(line[1:4]) / 5 
        var_list.append(var)
        # print(var_list)
        Hz_list.append(line[4] / 5)
        mv_list.append(line[0])

    plt.figure()
    plt.errorbar(mv_list, Hz_list, yerr = var_list, fmt='o')
    plt.yscale('log')
    plt.tick_params(labelsize = 13)
    plt.ylabel('rate(Hz)', fontsize = 15)
    plt.xlabel('threshold(mv)', fontsize = 15)
    plt.title('threshold vs rate', fontsize = 15)
    plt.show()
    # plt.savefig('./graph/rate.png')
