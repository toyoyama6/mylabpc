from kikusui import PMX70_1A
from subprocess import PIPE,Popen
import time
import numpy as np
import os

v_list = np.linspace(2.5, 3.3, 20)
v_list = [float(f'{i:.2f}') for i in v_list]
dcpower = PMX70_1A('10.25.123.249')
# i = 1
# while os.path.exists('./data/test' + str(i)) == False:
#     os.mkdir('./data/test' + str(i + 1))
#     break

for i in v_list:
    dcpower.change_volt_current(i, 0.02)
    time.sleep(10)
    filename = '{}V'.format(i)
    cmd = 'python3 read_waveform.py {0} {1} {2}'.format(1, './data/test15/' + filename, 500)
    proc = Popen(cmd, stdout=PIPE,shell=True)
    proc.communicate()
 
