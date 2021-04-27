import numpy as np
from src import network, utils
import matplotlib.pyplot as plt
from src.utils import *




ts_train = np.arange(0, t_max, utils.dt)


def normal(x, mu, sigma):
    normal = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

    return normal

def two_normals(t, m1=25, m2=100, s1=20, s2=10):
    return 10*(normal(t, m1, s1) + normal(t, m2, s2))

two_norm = np.vectorize(two_normals)

net = network.EchoState(target_f=two_norm)

## Start Training phase
change_list = ['J', 'u', 'w']
for i in range(int(t_max//dt)):

    net.step(change=change_list)

# z_list = np.asarray(net.z_list['train'])
# print(z_list.shape)

plt.plot(ts_train, two_norm(ts_train), label='Target Function')
plt.plot(*zip(*net.z_list['train']), label='$z$')

if 'w' in change_list:
    plt.plot(*zip(*net.dw_list), label='$|dw|$')
if 'J' in change_list:
    plt.plot(*zip(*net.dJ_list), label='$|dJ|$')
plt.legend()
plt.show()
