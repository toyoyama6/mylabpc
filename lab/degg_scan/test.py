import numpy as np



df = np.load('./data/20220329_00/total_data.npz')

print(df)
print(df['radius'])
print(df['phi'])
print(df['val'])