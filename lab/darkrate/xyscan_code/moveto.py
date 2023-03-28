import thorlabs_apt as apt

import sys
import time



argv = sys.argv
argc = len(argv)

if(argc!=3):
	exit(1)

to_x, to_y = 0.0, 0.0

fnow = open("currentpos.txt", mode="r")
lastline = fnow.readlines()[-1]
fnow.close()

x, y = lastline.split("\t")

x = float(x)
y = float(y)

try:
	to_x = float(argv[1])
	to_y = float(argv[2])
except:
	exit(1)


apt.list_available_devices()
motor_y = apt.Motor(45188554)
motor_x = apt.Motor(45871697)

motor_x.move_to(to_x)
time.sleep(1)
motor_y.move_to(to_y)

fout = open("currentpos.txt", mode="a")
fout.write("{0}\t{1}\n".format(str(to_x),str(to_y)))
fout.close()
print("Current position is ({0},{1})".format(str(to_x), str(to_y)))

