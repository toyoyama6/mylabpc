import click
import os
import sys
from tqdm import tqdm
import glob
import natsort

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_xscan import *


def plot_2d_heatmap(x, y, val, graph_dir, top_bottom, mean_std='mean'):

    X, Y = np.meshgrid(y, x)

    plt.figure()
    plt.title(f'{top_bottom}-z scan: relative sensitivity heatmap', fontsize=18)
    plt.ylabel('z (mm)', fontsize=16)
    plt.xlabel('phi (degree)', fontsize=16)
    plt.grid()
    plt.pcolormesh(X, Y, val.T)
    plt.colorbar()
    plt.savefig(f'{graph_dir}{top_bottom}_relative_{mean_std}_charge_map_yscan.png', bbox_inches='tight')
    plt.close()


def plot_phi_efficiensy(theta_range, relative_mean_charge, z_range, graph_dir, theta_range_deg):

	plt.figure()
	plt.title('phi vs relative sensitivity', fontsize=18)
	plt.xlabel('z (mm)', fontsize=16)
	plt.ylabel('relative sensitivity', fontsize=16)
	plt.grid()
	for i, phi in enumerate(theta_range):
		y = relative_mean_charge[i]
		plt.scatter(z_range, y, color=toColor(theta_range_deg[i], theta_range_deg))
		plt.plot(z_range, y, color=toColor(theta_range_deg[i], theta_range_deg), label=f'phi = {phi}')
		np.savez(f'{graph_dir}{phi}', z=z_range, charge=y)
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.savefig(f'{graph_dir}relative_charge_every_phi.png', bbox_inches='tight')
	plt.close()


def analysis_yscan(file_name, graph_dir, top_bottom, ref_data_dir):

    ##scan data
    df = pd.read_hdf(file_name, 'df')
    theta_range = np.array(df.t_point.unique())
    theta_range = np.sort(theta_range)
    z_range = np.array(df.r_point.unique())
    z_range = np.sort(z_range)
    mean_charge = []
    std_charge = []

    for i in tqdm(theta_range):
        data = df[df.t_point == i]
        mean_charge_theta, std_charge_theta = point_hist(data, i, z_range, graph_dir)
        mean_charge.append(mean_charge_theta)
        std_charge.append(std_charge_theta)

    mean_charge = np.array(mean_charge)
    std_charge = np.array(std_charge) 
    theta_range_rad = np.deg2rad(theta_range)

    relative_mean_charge = mean_charge/np.max(mean_charge)
	
    if(top_bottom=='t'):
        z_range = -1*z_range + 293
    elif(top_bottom=='b'):
        z_range = z_range - 293

    plot_2d_heatmap(z_range, theta_range, relative_mean_charge, graph_dir, top_bottom, mean_std='mean')
    plot_2d_heatmap(z_range, theta_range, std_charge, graph_dir, top_bottom, mean_std='std')

    dfs = glob.glob(f'{ref_data_dir}*hdf5')
    dfs = natsorted(dfs)

    strthetas = []
    ref_charge = []
    ref_charge_error = []
    for i in dfs:
        strtheta = float(i.split('.hdf5')[0].split('/')[-1].split('_')[-1])
        mean, std = get_reference_data(i)
        strthetas.append(strtheta)
        ref_charge.append(mean)
        ref_charge_error.append(std/10)
    plot_ref_stability(strthetas, ref_charge, ref_charge_error, graph_dir)


    plot_phi_efficiensy(theta_range, relative_mean_charge, z_range, graph_dir, theta_range_rad)
    
    np.savez(f'{graph_dir}total_data', height=z_range, phi=theta_range_rad, val=relative_mean_charge) 

##################################################
@click.command()
@click.argument('data_dir')
@click.argument('top_bottom')
@click.argument('graph_dir_name')
def main(data_dir, top_bottom, graph_dir_name):

    graph_dir = '/home/yuya_takemasa/degg_scan/degg_scan/graph/'+graph_dir_name+'/'
    file_name = f'{data_dir}sig/charge_stamp.hdf5'
    ref_data_dir = f'{data_dir}ref/'
    try:
        os.mkdir(graph_dir)
    except:
        ans = input('Overwrite ??? (y/n): ')
        if(ans=='y'):
            print('OK!!')
        else:
            sys.exit()
    analysis_yscan(file_name, graph_dir, top_bottom, ref_data_dir)
if __name__ == '__main__':
    main()
##end
