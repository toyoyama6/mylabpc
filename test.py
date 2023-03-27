import numpy as np
import pandas as pd





def gaussian(x, Aped, Mped, Sped):
    return Aped * np.exp(-(x - Mped) ** 2 / (2 * Sped ** 2))

def fit_gaussian(data):
    x, y = _hist_to_coords(data)
    mean_index = np.argmax(y)
    Aped = y[mean_index]
    Mped = x[mean_index]
    Sped = 20
    p0 = [Aped, Mped, Sped]
    popt, pcov = curve_fit(gaussian, x, y, p0 = p0, maxfev=4000)
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

r_step = 3 ##mm
r_max = 141 ##mm (radius)
r_range = np.arange(0, r_max, r_step)

z_step = 3 ##mm
z_max = 135 ##mm (radius)
z_range = np.arange(0, z_max, z_step)

theta_step = 6 ##deg
theta_max = 360 ##deg
theta_range = np.arange(0, theta_max, theta_step)

theta_length = len(theta_range)
r_length = len(r_range)
hitmap = np.zeros((theta_length, r_length), dtype=np.float32)
np.add.at(hitmap, (3, 4), 10)
data_dir = './'
for z_num, z in enumerate(z_range): 
	for r_num, r in enumerate(r_range):
		file = "{}/df_matched_trigger_{}_{}.hdf".format(data_dir, str(r), str(z))
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
			np.add.at(hitmap, (z_num, r_num), mean)

		except:
			np.add.at(hitmap, (z_num, r_num), -1)


print(hitmap)
