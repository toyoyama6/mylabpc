import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate
from tqdm import tqdm
import glob

volt_list = np.linspace(2.5, 3.3, 20)
volt_list = [float(f'{i:.2f}') for i in volt_list]
delta_t = (6.404 * 10 ** -7 - -5.592 * 10 ** -7) / 3000
t =  np.arange(-5.592 * 10 ** -7, 6.404 * 10 ** -7, delta_t)
test_list = ['fiber1', 'fiber2', 'fiber2_2', 'test15']

for test in test_list:
    charge_list_mean = []
    for v in tqdm(volt_list):   
        readname = './data/' +  test + '/' + str(v) + 'V_ch0' + '.txt'
        charge_list = []
        data = np.loadtxt(readname, skiprows=1)
        data = -data

        for i in range(500):
            data[i] = data[i] - np.mean(data[i][:50])
            # plt.plot(data[i])
            # plt.show()            
            value = integrate.simps(data[i][1650:1850], t[1650:1850]) / 10000
            charge = (value / 50) * 10 ** 12 #pc
            charge_list.append(charge)
        
        charge_list_mean.append(np.mean(charge_list)) 

        # plt.hist(charge_list, bins = 30)
        # plt.show()

    plt.scatter(volt_list, charge_list_mean, label = f'{test}')    

plt.title('pmt at fiber1')
# plt.xticks(volt_list, [f'{volt}V' for volt in volt_list], v_mean)
plt.xlabel('volt')
plt.ylabel('charge[pc]')
plt.legend()
plt.savefig('./graph/fiber12.png')
plt.show()