import thorlabs_apt as apt


apt.list_available_devices()
motor_y = apt.Motor(45188554)
motor_y.move_home(True)

motor_x = apt.Motor(45871697)
motor_x.move_home(True)

fout = open("currentpos.txt", mode="a")
fout.write("0.0\t0.0\n")
fout.close()



