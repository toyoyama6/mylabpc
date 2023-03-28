import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate
from tqdm import tqdm

distance_list = ['0.5','3.0', '4.5']
filter_list = ['005', '010', '025', '032', '050', '100']
detector = input('detector:')
x0 = (6.576 * 10 ** -7 - -5.420 * 10 ** -7) / 3000
x1 =  np.arange(-5.420 * 10 ** -7, 6.576 * 10 ** -7, x0)
v_mean1 = []
x = [5, 10, 25, 32, 50, 100]

if detector == 'mppc':
    distance = input('distance: ')    
    for filter in tqdm(filter_list):
          readname = './data/' + detector + '_' + distance + '_' + filter +'_ch0' + '.txt'
          v_list1 = []
          for i in tqdm(range(100)):
               df = open(readname, 'r')
               data = df.readlines()[i+1]
               data = data.split()
               data = [float(x) for x in data]
               y = data - np.mean(data[:50])      
               value = integrate.simps(y[1400:2600], x1[1400:2600])
               charge = (value / 50) * 10 ** 12
               v_list1.append(charge)
               df.close()
          # # plt.hist(v_list1,bins = 30, range=(np.min(v_list1), np.max(v_list1)))
          # # plt.title('integerade  total = 100events')
          # # plt.show()
          v_mean1.append(np.mean(v_list1))           

    y = v_mean1 / v_mean1[0] 
    plt.figure()
    plt.title(f'mppc distance={distance}')
    plt.xlabel('filter[%]')
    plt.ylabel('charge[pc]')
    plt.scatter(x, y)    
    plt.xticks(x, ['5%', '10%', '25%', '32%', '50%', '100%'])
    plt.show()


elif detector == 'pmt':
    # plt.figure()
    # plt.title('pmt')
    # x = [5, 10, 25, 32, 50, 100]
    # plt.xlabel('filter[%]')
    # plt.ylabel('charge[pc]')
    # plt.xticks(x, ['5%', '10%', '25%', '32%', '50%', '100%'])
    for distance in distance_list:
        v_mean2 = []
        for filter in tqdm(filter_list):
            readname = './data/' + detector + '_' + distance + '_' + filter +'_ch0' + '.txt'
            v_list2 = []
            for k in range(100):
                df = open(readname, 'r')
                data = df.readlines()[k+1]
                data = data.split()
                data = [-float(x) for x in data]
                y = data - np.mean(data[:50])
                # print(y)
                # plt.plot(y)        
                value = integrate.simps(y[1100:1800], x1[1100:1800])
                charge = (value / 50) * 10 ** 12
                v_list2.append(charge)
                df.close()
            # plt.title(f'distance = {distance}cm  filter = {filter}%')
            # plt.show()
            # print(v_list2)
            # plt.hist(v_list2,bins = 30, range=(np.min(v_list2), np.max(v_list2)))
            # plt.title('integerade total = 100events')
            # plt.show()
            mean = np.mean(v_list2)
            v_mean2.append(np.mean(v_list2))

        # print(v_mean2)
        v_mean3 = v_mean2 / v_mean2[0]
        plt.scatter(x,v_mean3, label = f'distance = {distance}cm')
    plt.title('compared to 5%')
    plt.legend()
    plt.show()

    

