import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import sys
import time
from scipy import integrate
from tqdm import tqdm
import glob



def Get_charge(data, t, start, end):
	charge_list = []
	peak_list = []
	for wf in data:
		wf = -wf
		wf = wf - np.mean(wf[:250])
		#plt.ylabel('amplitude [mV]', fontsize = 25)
		#plt.xlabel('time [ns]', fontsize = 25)
		# plt.xlim(80,180) 
		#plt.plot(wf)
		#plt.tight_layout()
		#plt.show()
		charge = integrate.simps(wf[start:end], t[start:end]) / 1000 / 50 * 1e12
		charge_list.append(charge)
	#plt.savefig('test.png')
		#plt.show()
	return charge_list

def main():
	graph_dir = './graph/nakamura/gelpad/'
	data_dir = './data/nakamura/gelpad/2022-12-08-1345_55/'
	t_start = -3.068 * 10 ** (-7)
	t_end = 5.952 * 10 ** (-7)
	datapoints = 2256
	delta_t = (t_end - t_start) / datapoints
	t =  np.arange(t_start, t_end, delta_t)
	file_list = glob.glob(data_dir + '*txt')

	value_list = []
	for file in tqdm(file_list):
		x = file.split('/')[-1].split('_')[0]
		y = file.split('/')[-1].split('_')[1]
		print(file.split('/'))
		print(x,y)
		data = np.loadtxt(file, skiprows=1)
		charge_list = Get_charge(data, t, 1000, 1125)
		value_list.append([float(x), float(y), np.mean(charge_list)])

	df = pd.DataFrame(data = value_list, columns=["x", "y", "charge"])
	df.to_hdf(data_dir + '/charge.hdf5', key = 'df')

if __name__ == '__main__':
	main()
