import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from tqdm import tqdm

x = symbols('x')
E = symbols('E')

phi = (-1 / 1.3) * ((10 ** 7) ** (-1.3) - (10 ** 4) ** (-1.3))
a = 1 / phi

phi = a * E ** (-2.3) 
c = integrate(phi, (E, 10 ** 4, x))


random = np.random.rand(10**7)


ans_list = []
for i in tqdm(random):
    ans = ((1.00012590839211 - i) / 158509.274381464) ** (-1 / 1.3)
    # ans = solve(i - c, x)[0]
    #print(type(ans))
    ans_list.append(float(ans))

ans_list = [np.log10(i) for i in ans_list]
#3
# prob_list = []
# for j in ans_list:
#     prob = 10 ** (-5) * (j / (10 ** 4)) ** (0.35)
    # prob_list.append(prob)
# print(prob_list)

# 1,2
# t = np.linspace(10 ** 4, 10 ** 7, 100000)
# plt.figure()
# plt.title('phi', fontsize = 20)
# plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel('E', fontsize = 16)
# plt.ylabel('phi', fontsize = 16)
# plt.plot(t, a * t ** (-2.3))
# plt.tight_layout()
# # plt.savefig('./graph/neutrinos_energy_distribution.png')
# plt.show()
# plt.close 



# # 1,2
# plt.figure()
# plt.title('cumelative function', fontsize = 20)
# plt.xscale("log")
# plt.xlabel('x', fontsize = 16)
# plt.ylabel('C(x)', fontsize = 16)
# plt.plot(t, 1.00012590839211 - 158509.274381464 * t ** (-1.3))
# plt.tight_layout()
# # plt.savefig('./graph/cumelative_function.png')
# plt.show()
# plt.close 

# 1,2
plt.figure()
plt.title('number of neutrinos', fontsize = 20)
# plt.xscale("log")
plt.xlabel('log10(E)', fontsize = 16)
plt.ylabel('events', fontsize = 16)
plt.hist(ans_list)
plt.tight_layout()
plt.savefig('./graph/number_of_neutrinos.png')
plt.show()
plt.close 

