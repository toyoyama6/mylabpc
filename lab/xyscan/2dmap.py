import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



file = './data/nakamura/gelpad/2022-12-08-1345_55/charge.hdf5'
graph_dir = './graph/nakamura/'
df = pd.read_hdf(file)
x_range = np.array(sorted(df.x.unique()))
y_range = np.array(sorted(df.y.unique())) 
print('y_range = ', y_range)
y_cut = 80
q_sum = df.charge.sum()

mean_charge = []
cutting_list = []
for y in y_range:	
	data = df[df.y == y]
	data = data.sort_values(['x'], ascending=True)
	val = data.charge
	mean_charge.append(val)
	if y == y_cut:
		cutting_list.append(val)


X, Y = np.meshgrid(x_range, y_range)
mean_charge = np.array(mean_charge)
mean_charge = mean_charge / np.max(mean_charge)

plt.figure()
plt.pcolormesh(X, Y, mean_charge)
# plt.ylim(-100, 100)
# plt.xlim(-100, 100)
#plt.gca().set_aspect('equal', adjustable='box')
# plt.title(f'{degree_list[i]}' + 'degree', fontsize = 20)
plt.xlabel('x', fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.colorbar().set_label('charge[pc]', fontsize = 18)
plt.savefig(graph_dir + './2dmap.png', bbox_inches = 'tight')
#plt.show() 
plt.close()


plt.figure()
plt.title('y = {}'.format(y_cut), fontsize = 20)
plt.ylabel('relative charge[pC]', fontsize = 18)
plt.xlabel('x[mm]', fontsize = 18)
plt.scatter(x_range, cutting_list)
plt.savefig(graph_dir + 'cutting_plot.png')
plt.show()
