import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm 
import scipy
import click
import os


def plot_2d_heatmap(z, theta, val, graph_dir, meas_type, mean_std):

    X, Y = np.meshgrid(theta, z)
    plt.figure()
    plt.title(f'{meas_type} scan: relative sensitivity heatmap', fontsize=18)
    plt.ylabel('z (mm)', fontsize=16)
    plt.xlabel('phi (degree)', fontsize=16)
    plt.grid()
    plt.pcolormesh(X, Y, val.T)
    plt.colorbar()
    plt.savefig(f'{graph_dir}{meas_type}_relative_charge.png', bbox_inches='tight')
    plt.close()

def plot_polar_heatmap(r, theta, val, graph_dir, meas_type, mean_std):

    X, Y = np.meshgrid(theta, r)

    plt.figure()
    plt.subplot(projection="polar")
    plt.pcolormesh(X, Y, val.T)
    plt.grid()
    plt.title(f'{meas_type} scan: relative sensitivity heatmap', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.colorbar()
    plt.savefig(f'{graph_dir}{meas_type}_relative_{mean_std}_charge.png', bbox_inches='tight')
    plt.close()

def analysis_wrapper(file, meas_type, graph_dir):
    df = pd.read_hdf(file)
    df = df.where(df > 0, np.nan)
    df = df.interpolate(limit_direction="both")
    df = df.values
    print(np.max(df))
    # df = df / np.max(df)
    print(np.shape(df))
    theta_step = 6 ##deg
    theta_max = 360 ##deg
    theta_range = np.arange(0, theta_max, theta_step)
    if meas_type == "top-r" or meas_type == "bottom-r":
        r_step = 3 ##mm
        r_max = 141 ##mm (radius)
        r_range = np.arange(0, r_max, r_step)
        theta_range = np.deg2rad(theta_range)
        plot_polar_heatmap(r_range, theta_range, df, graph_dir, meas_type, mean_std = "mean")
    if meas_type == "top-z" or meas_type == "bottom-z":
        r_step = 3 ##mm
        r_max = 135 ##mm (radius)
        r_range = np.arange(0, r_max, r_step)
        plot_2d_heatmap(r_range, theta_range, df, graph_dir, meas_type, mean_std = "mean")




@click.command()
@click.argument('data_dir')
def main(data_dir):
    file = data_dir + "sig/trigger/2d_data.hdf"
    deggid = file.split("/")[-6]
    meas_type = file.split("/")[-5]
    date = file.split("/")[-4]
    print(deggid)
    print(meas_type)
    print(date)
    if(not os.path.exists(f'./figs/{deggid}/')):
        os.mkdir(f'./figs/{deggid}/')
    if(not os.path.exists(f'./figs/{deggid}/{meas_type}/')):
        os.mkdir(f'./figs/{deggid}/{meas_type}/')
    if(not os.path.exists(f'./figs/{deggid}/{meas_type}/{date}/')):
        os.mkdir(f'./figs/{deggid}/{meas_type}/{date}/')
    graph_dir = f"./figs/{deggid}/{meas_type}/{date}/"
    print(deggid)
    print(meas_type)
    print(graph_dir)
    analysis_wrapper(file, meas_type, graph_dir)


if __name__ == "__main__":
    main()