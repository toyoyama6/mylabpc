import thorlabs_apt as apt

import sys
import time


to_x, to_y = 0.0, 0.0

fnow = open("currentpos.txt", mode="r")
lastline = fnow.readlines()[-1]
fnow.close()

x, y = lastline.split("\t")

x = float(x)
y = float(y)

print(x,y)


apt.list_available_devices()
motor_y = apt.Motor(45188554)
motor_x = apt.Motor(45871697)


fout = open("currentpos.txt", mode="a")

delx = 20.0
dely = 20.0


motor_x.move_to(x-delx/2, True)
time.sleep(1)
motor_y.move_to(y-dely/2, True)

x += -delx/2
y += -dely/2

stepsize = 11

stepx = delx/stepsize
stepy = dely/stepsize

fout.write("{0}\t{1}\n".format(str(x),str(y)))
print("Now.. ({0}, {1})".format(str(x),str(y)))


for ix in range(stepsize):
	for iy in range(stepsize):
		y += stepy
		motor_y.move_to(y, True)
		print("Now.. ({0}, {1})".format(str(x),str(y)))
		fout.write("{0}\t{1}\n".format(str(x),str(y)))
		time.sleep(0.5)
	x += stepx
	motor_x.move_to(x, True)
	print("Now.. ({0}, {1})".format(str(x),str(y)))
	fout.write("{0}\t{1}\n".format(str(x),str(y)))
	time.sleep(0.5)
	stepy *= -1

x += -delx/2
y += -dely/2
motor_x.move_to(x, True)
motor_y.move_to(y, True)

fout.write("{0}\t{1}\n".format(str(x),str(y)))
fout.close()

print("Current position is ({0},{1})".format(str(x), str(y)))

