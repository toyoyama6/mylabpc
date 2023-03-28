import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

data_dir = './dark_data/ampli_data/'

df = ['-36.csv', '-44.5.csv', '-9.csv', 'room.csv']
df1 = ['20212100_-10_ch1-10.csv']
df2 = ['20212200_room_ch123.csv', '20210709-0800_-12_ch1-12.csv']

tmp = [-36, -44.5, -9, 23]
tmp1 = [-10]
tmp2 = [20, -12]
ave_nlate = []
std_nlate = []

num = 0
for i in df:
    ddf = pd.read_csv(data_dir+i)
    print(data_dir+i)
    x = ddf['nlate']
    ave_nlate.append(np.mean(x))
    std_nlate.append(np.std(x))
    plt.figure()
    plt.hist(x, bins=15, range=(-0.5, 14.5))
    plt.title('Number of after-pulse T={}'.format(tmp[num]))
    plt.xlabel('N of after-pulse')
    plt.yscale('log')
    plt.ylim(0, 1000)
    plt.savefig('./dark_graph/N_of_afterpulse-hist-T_{}.png'.format(tmp[num]))
    plt.close()

    num += 1

numm = 0

ave_nlate1 = []
std_nlate1 = []

for i in df1:
    ddf1 = pd.read_csv(data_dir+i)
    print(data_dir+i)
    x = ddf1['nlate']
    ave_nlate1.append(np.mean(x))
    std_nlate1.append(np.std(x))
    plt.figure()
    plt.hist(x, bins=15, range=(-0.5, 14.5))
    plt.title('Number of after-pulse T={}'.format(tmp1[numm]))
    plt.xlabel('N of after-pulse')
    # plt.ylim(0, 10000)
    plt.yscale('log')
    plt.ylim(0, 1000)
    plt.savefig('./dark_graph/N_of_afterpulse-hist-T_new_{}.png'.format(tmp1[numm]))
    plt.close()

    numm += 1

nummm = 0

ave_nlate2 = []
std_nlate2 = []

for i in df2:
    ddf2 = pd.read_csv(data_dir+i)
    print(data_dir+i)
    x = ddf2['nlate']
    ave_nlate2.append(np.mean(x))
    std_nlate2.append(np.std(x))
    plt.figure()
    plt.hist(x, bins=15, range=(-0.5, 14.5))
    plt.title('Number of after-pulse T={}'.format(tmp2[nummm]))
    plt.xlabel('N of after-pulse')
    # plt.ylim(0, 10000)
    plt.yscale('log')
    plt.ylim(0, 1000)
    plt.savefig('./dark_graph/N_of_afterpulse-hist-T_mdom_{}.png'.format(tmp2[nummm]))
    plt.close()

    nummm += 1

print(ave_nlate, ave_nlate1, ave_nlate2)


plt.figure()
plt.title('average after-pulse per triggered signal')
plt.xlabel('Tenperature ({}C)'.format(chr(176)))
plt.ylabel('average after-pulse per triggered signal')
plt.scatter(tmp, ave_nlate, c='r', s=30, label='old method')
plt.scatter(tmp1, ave_nlate1, c='black', s=30, label='new method')
# plt.scatter(tmp2, ave_nlate2, c='b', s=30, label='new method_mdom')
plt.ylim(-0.1, 2)
plt.xlim(-48, 28)
plt.legend()
plt.tight_layout()

plt.savefig('./dark_graph/N_of_afterpulse_old_new.png')
plt.close()
# plt.show()
