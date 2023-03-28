import os
import glob 
from natsort import natsorted



file_list = glob.glob('./data/BB9759/-40/darkrate/try3/*.npz')
file_list = natsorted(file_list)

for file in file_list:
    size = os.path.getsize(file) / 1000
    print(size)
    # if size > 30:
    #     print(size, 'kB')
