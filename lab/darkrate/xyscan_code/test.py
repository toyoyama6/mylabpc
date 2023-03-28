import thorlabs_apt as apt

import sys
import time


from subprocess import PIPE,Popen


to_x, to_y = 0.0, 0.0


fnow = open("currentpos.txt", mode="r")
lastline = fnow.readlines()[-1]
fnow.close()

x, y = lastline.split("\t")

x = float(x)
y = float(y)

print(x,y)

delx = 7.0
dely = 7.0

x += -delx/2
y += -dely/2

stepsize = 9

stepx = delx/stepsize
stepy = dely/stepsize

print("Now.. ({0}, {1})".format(str(x),str(y)))

nwfm = 50
for ix in range(stepsize):

	for iy in range(stepsize):
		y += stepy
		print("Now.. ({0}, {1})".format(str(x),str(y)))

	x += stepx
	print("Now.. ({0}, {1})".format(str(x),str(y)))
	stepy *= -1

for iy in range(stepsize):
	y += stepy
	print("Now.. ({0}, {1})".format(str(x),str(y)))

x += -delx/2
y +=  dely/2

print("Current position is ({0},{1})".format(str(x), str(y)))

