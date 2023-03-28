import uproot
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
uproot.default_library="np"
import math


radius = 100

fQE_LOM  = np.loadtxt("QE_4inch.txt", delimiter=",")
lam_LOM = fQE_LOM[:,0]
QE_LOM = fQE_LOM[:,1]/100.0

itp_QE_LOM  =  interpolate.interp1d(lam_LOM, QE_LOM)

CEdata = np.loadtxt("CE_4inch_r.txt")
r   = CEdata[:,0]
CE  = CEdata[:,1]

itp_CE  =  interpolate.interp1d(r, CE)


plt.figure(figsize=(9,7))
plt.grid()

nbinx = 100
nbiny = 100
nbinang = 19
half_x = radius
half_y = radius

xedge = np.linspace(-half_x, half_x, nbinx+1)
yedge = np.linspace(-half_y, half_y, nbiny+1)

xc = 0.5*(xedge[1:]+xedge[:-1])
yc = 0.5*(yedge[1:]+yedge[:-1])

angs = np.linspace(0, np.pi, nbinang)


datafile = "./data/singlePMT_wtopgelpad.root"
file = uproot.open(datafile)
tr = file["tree"].arrays(library="np")

wavelength = tr["wavelength"]
angle      = np.pi-tr["angle"]
hitpos     = tr["LocalHitPos"]
modID      = tr["ModID"]
genpos     = tr["GenPos"]

x = np.cos(angle)*genpos[:,0] - np.sin(angle)*genpos[:,2]
y = genpos[:,1] 


angle_set_list = sorted(list(set(angle)))

iang = np.int32((angle+0.001)*180/np.pi/10)

hitmap = np.zeros((nbinang, nbinx, nbiny), dtype=np.float32)

r = np.sqrt(hitpos[:,0]*hitpos[:,0]+hitpos[:,1]*hitpos[:,1])
CEQE = itp_CE(r)*itp_QE_LOM(400.0)

ix = np.int32(nbinx*(x+half_x)/2/half_x)
iy = np.int32(nbiny*(y+half_y)/2/half_y)

np.add.at(hitmap, (iang,ix,iy), CEQE)

rel_LY = np.average(hitmap, axis=(1,2))
print(hitmap)