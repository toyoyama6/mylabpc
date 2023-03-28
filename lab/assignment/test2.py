import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

 

 

 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
phi = (-1 / 1.3) * ((10 ** 7) ** (-1.3) - (10 ** 4) ** (-1.3))
a = 1 / phi
print(a)

 
def hist(title, xlabel,ylabel,  data, bins, save_name):
    plt.figure()
    plt.title(title, fontsize = 20)
    # plt.yscale("log")
    plt.xlabel(xlabel, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 16)
    plt.hist(data, bins = bins.format(bins))
    plt.tight_layout()
    # plt.savefig('./graph/' + save_name)
    plt.show()
    plt.close

 

def energy(r):
    # ans_list = []
    # for i in tqdm(random):
    E = ((1.00012590839211 - r) / 158509.274381464) ** (-1 / 1.3)
    return E
    # ans = solve(i - c, x)[0]
    #print(type(ans))
    # ans_list.append(float(ans))
    # return ans_list

def probability(E):
    # prob_list = []
    # for j in ans_list:
    prob = 10 ** (-5) * (E / (10 ** 4)) ** (0.35)
        # prob_list.append(prob)
    return  prob

 

e_list = []
energy_list = []
random = np.random.rand(10 ** 7)

for r in tqdm(random):
    e = energy(r)
    e_list.append(e)
    p = probability(e)
    choice = np.random.choice(['True', 'False'], 1, p=[p, 1 - p])
    # print(choice)
    if choice == 'True':
        energy_list.append(e)
    else:
        continue


# print(energy_list)

 

loge_list = [np.log10(i) for i in e_list]
logenergy_list = [np.log10(i) for i in energy_list]

 


1,2
t = np.linspace(10 ** 4, 10 ** 7, 100000)
plt.figure()
plt.grid()
plt.title('phi', fontsize = 20)
plt.xscale("log")
# plt.yscale("log")
plt.xlabel('E', fontsize = 16)
plt.ylabel('phi', fontsize = 16)
plt.plot(t, a * t ** (-2.3))
plt.tight_layout()
plt.savefig('./graph/neutrinos_energy_distribution.png')
plt.show()
plt.close

 


 
# 1,2
plt.figure()
plt.grid()
plt.title('cumulative function', fontsize = 20)
plt.xscale("log")
plt.xlabel('x', fontsize = 16)
plt.ylabel('C(x)', fontsize = 16)
plt.plot(t, 1.00012590839211 - 158509.274381464 * t ** (-1.3))
plt.tight_layout()
plt.savefig('./graph/cumulative_function.png')
plt.show()
plt.close

 

#1,2
plt.figure()
plt.title('1-2 number of neutrinos', fontsize = 20)
# plt.yscale("log")
plt.xlabel('log10(E)', fontsize = 16)
plt.ylabel('events', fontsize = 16)
plt.hist(loge_list, bins = 50)
plt.tight_layout()
plt.savefig('./graph/hist_1e7neutrinos.png')
plt.show()
plt.close

 

# #3
# plt.figure()
# plt.title('number of neutrinos', fontsize = 20)
# # plt.yscale("log")
# plt.xlabel('log10(E)', fontsize = 16)
# plt.ylabel('events', fontsize = 16)
# plt.hist(logenergy_list)
# plt.tight_layout()
# # plt.savefig('./graph/hist_10neutrinos.png')
# plt.show()
# plt.close

 