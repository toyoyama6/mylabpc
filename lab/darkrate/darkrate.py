import glob
import numpy as np
import matplotlib.pyplot as plt

 
data_dir = './data/BB9759/25/darkrate/try2/'
# file_list = glob.glob(data_dir + '*.txt')
# file = data_dir + 'data_2022_04_19_17_59_04.txt'
file = data_dir + 'data_2022_04_19_20_45_49.txt'
# print(file_list)
# print(len(file_list))
peakH = []
charge = []
deltaT = []



data = np.loadtxt(file, skiprows=1)       
for i in range(len(data)):
    # print(data[:,2][i]) 
    peakH.append(data[:,4][i])
    charge.append(data[:,3][i])
    deltaT.append(data[:,2][i])


deltaT = [i for i in deltaT if i != -1]
# hist1, bins1 = np.histogram(deltaT, bins = 30)
# print(hist1)
Hz = [1 / i  for i in deltaT]  #kHz
# print(Hz)

log10_deltaT = [np.log10(i) for i in deltaT]
# print(len(log10_deltaT))
time_list = []
time = 0
times = []


for i in deltaT:
    time += i
    time_list.append(time)
# print(time_list)
for i in range(len(deltaT)):
    times.append(i)

plt.figure()
plt.hist(log10_deltaT, bins = 20)
plt.yscale('log')
plt.title('deltT')
# plt.xlabel(f'{xlabel}')
# plt.ylabel(f'{ylabel}')
# plt.savefig('./peak.png')
# plt.show()
plt.close()


plt.figure()
plt.yscale('log')
plt.xlabel('charge[pc]', fontsize = 16)
plt.ylabel('events', fontsize = 16)
plt.title('charge histgram', fontsize = 18)
plt.hist(charge, bins = 100)
plt.tight_layout()
plt.savefig('./graph/BB9759/25/darkrate/charge_histgram_before.png')
# plt.show()
plt.close()

plt.figure()
plt.yscale('log')
plt.scatter(times, Hz, s = 0.5)
plt.title('rate')
# plt.show()
plt.close()

plt.figure()
plt.yscale('log')
plt.title('peakH')
plt.hist(peakH)
# plt.show()
plt.close()
