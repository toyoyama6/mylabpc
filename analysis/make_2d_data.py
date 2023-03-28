import numpy as np
import pandas as pd
import click
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate
from tqdm import tqdm
import tables
from natsort import natsorted
import math
from scipy.interpolate import interp1d
from tqdm import tqdm
from glob import glob




def gaussian(x, Aped, Mped, Sped):
    return Aped * np.exp(-(x - Mped) ** 2 / (2 * Sped ** 2))

def fit_gaussian(data):
    x, y = _hist_to_coords(data)
    mean_index = np.argmax(y)
    Aped = y[mean_index]
    Mped = x[mean_index]
    Sped = 20
    p0 = [Aped, Mped, Sped]
    popt, pcov = curve_fit(gaussian, x, y, p0 = p0)
    return popt



def _hist_to_coords(data, nbin = 100):

    density, bin_edges = np.histogram(data, bins = np.linspace(0, max(data), nbin))
    widths = bin_edges[1:] - bin_edges[:-1]
    centres = (bin_edges[1:] + bin_edges[:-1]) / 2
    x = centres
    y = density
    f = interp1d(centres, density)
    x = np.linspace(centres.min(), centres.max(), nbin * 100, endpoint = False)
    y = f(x)
    return x, y

def plot_hist(charge, r_point, t_point, graph_dir, popt, n_bin = 100,  min_bin = 0, max_bin = 250):

    xd = np.arange(np.min(charge), np.max(charge), 0.01)
    estimated_curve = gaussian(xd, popt[0], popt[1], popt[2])

    plt.figure()
    plt.title(f'phi={t_point}:r={r_point} charge distribution', fontsize=18)
    plt.xlabel ('charge [pC]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.xlim(min_bin, max_bin)
    plt.hist(charge, bins=n_bin, range=(min_bin,max_bin), color='blue')
    plt.plot(xd, estimated_curve, color='red')
    plt.savefig(f'{graph_dir}{t_point}_{r_point}.png', bbox_inches='tight')
    # plt.show()
    plt.close()



def get_reference_data(filename):

    charges = []
    with tables.open_file(filename) as f:
        data = f.get_node('/data')
        waveforms = data.col("waveform")
        times = data.col("time")

        start_point = int(len(times[0])/2)
        end_point = int(start_point + start_point//3)
        base = np.mean(waveforms[0][0:200])

        for time, waveform in tqdm(zip(times, waveforms)):

            waveform = -waveform[start_point:end_point] + base
            time = time[start_point:end_point]
            charge = integrate.simps(waveform, time)/50*1e12
            charges.append(charge)

    popt = fit_gaussian(charges)
    return popt[1], popt[2]

def plot_ref_stability(x, y, y_err, graph_dir):
    plt.figure()
    plt.title('Reference PMT charge', fontsize=18)
    plt.ylabel('charge (pC)', fontsize=16)
    plt.xlabel('#phi', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid()
    plt.ylim(np.min(y)-10, np.max(y)+10)
    plt.errorbar(x, y, yerr=y_err, fmt='o', markersize=5, ecolor='black', markeredgecolor='black', color='white', capsize=3)
    plt.savefig(f'{graph_dir}refpmt_stability.png', bbox_inches='tight')
    plt.close()


def analysis_ref(ref_data_dir, graph_dir):
        #reference PMT check
    dfs = glob(f'{ref_data_dir}*hdf5')
    dfs = natsorted(dfs)

    strthetas = []
    ref_charge = []
    ref_charge_error = []
    for i in dfs:
        strtheta = i.split('.hdf5')[0].split('/')[-1].split('_')[-1]
        mean, std = get_reference_data(i)
        strthetas.append(float(strtheta))
        ref_charge.append(mean)
        ref_charge_error.append(std/10)
    plot_ref_stability(strthetas, ref_charge, ref_charge_error, graph_dir)
    # np.savez(f'{graph_dir}total_data', radius=r_range, phi=theta_range_rad, val=relative_mean_charge)



def analysis(data_dir, graph_dir, meas_type):
    sig_data_dir = data_dir + "sig/trigger/"
    ref_data_dir = data_dir + "ref/"
    theta_step = 6 ##deg
    theta_max = 360 ##deg
    theta_range = np.arange(0, theta_max, theta_step)
    if meas_type == "top-r" or meas_type == "bottom-r":
        r_step = 3 ##mm
        r_max = 141 ##mm (radius)
        r_range = np.arange(0, r_max, r_step)
    elif meas_type == "top-z" or meas_type == "bottom-z":
        r_step = 3 ##mm
        r_max = 135 ##mm (radius)
        r_range = np.arange(0, r_max, r_step)

    theta_length = len(theta_range)
    r_length = len(r_range)
    print(r_range)
    hitmap = np.zeros((theta_length, r_length), dtype=np.float32)
    print(np.shape(hitmap))
    for theta_num, theta in enumerate(tqdm(theta_range)): 
        for r_num, r in enumerate(r_range):
            file = "{}df_matched_trigger_{}_{}.hdf".format(sig_data_dir, str(r), str(theta))
            try:
                df = pd.read_hdf(file)
                try:
                    df_degg = df[(df.type=="degg") & (df.channel == 1) & (df.valid == True)]
                except:
                    df_degg = df[(df.type=="degg") & (df.channel == 1)]
            
                charge = df_degg.charge
                popt  = fit_gaussian(charge)
                mean = popt[1]
                std = popt[2]
                plot_hist(charge, r, theta, graph_dir, popt, n_bin = 100,  min_bin = 0, max_bin = 250)
                np.add.at(hitmap, (theta_num, r_num), mean)

            except:
                print("hey")
                np.add.at(hitmap, (theta_num, r_num), np.nan)
    print(hitmap)
    analysis_ref(ref_data_dir, graph_dir)
    df = pd.DataFrame(hitmap)
    df.to_hdf(f"{data_dir}/sig/trigger/2d_data.hdf", key = "df")

@click.command()
@click.argument('data_dir')
def main(data_dir):
    deggid = data_dir.split("/")[-4]
    meas_type = data_dir.split("/")[-3]
    date = data_dir.split("/")[-2]
    graph_dir = f"./figs/{deggid}/{meas_type}/{date}/"
    print(graph_dir)
    print(meas_type)
    analysis(data_dir, graph_dir, meas_type)


if __name__ == "__main__":
    main()
