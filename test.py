import numpy as np
from src import network, utils

t_max = 200

net = network.EchoState()

def target_f():
    


ts_train, ts_test = np.arange(0, t_max, network.dt), np.arange(t_max, 2*t_max, network.dt)
lw_f, lw_z = 3, 1.5

# TRAIN Phase
f_train = network.f(ts_train)

fig = plt.figure(figsize=(15, 3*4))
j = 1

for i, t in enumerate(ts_train):
