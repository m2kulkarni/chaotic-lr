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
change_list = ['J', 'w']
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

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def plot_histogram(net):
    rates_ini = np.tanh(net.x_initial)
    rates_fin = np.tanh(net.x)
    rates = np.vstack([rates_fin, rates_ini ])
    plt.hist(rates.T, bins=20, histtype='barstacked')
#    plt.hist(rates_fin)
    print(rates)
    plt.show()

def plot_J(net):
    J_ini = net.J_GG_initial
    scale = np.sqrt(np.sum(J_ini**2))
    J_fin = net.J_GG
    J_times = np.divide(J_fin, J_ini)
    #J_fin = J_fin/np.max(J_fin)
#    J = np.vstack([J_ini, J_fin])
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    print(J_ini[net.conditioned_neurons])
    print(J_fin[net.conditioned_neurons])
    im1 = ax1.imshow(J_ini[net.conditioned_neurons].T, aspect='auto')
    im2 = ax2.imshow(J_fin[net.conditioned_neurons].T, aspect='auto')
#    im3 = ax3.imshow(J_times[net.conditioned_neurons].T, aspect='auto')
    fig.colorbar(im2, ax=ax2 )
    fig.colorbar(im1, ax=ax1 )
#    fig.colorbar(im3, ax=ax3)
    plt.show()

    print(np.sum(J_ini))
    print(np.sum(J_fin))
#    plt.hist(J_fin[net.conditioned_neurons[:1]].T, histtype='barstacked')
#    plt.hist(J_ini[net.conditioned_neurons[:1]].T, histtype='barstacked')
#    plt.show()

plot_histogram(net)
plot_J(net)
