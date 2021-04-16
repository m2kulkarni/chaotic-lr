import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from copy import deepcopy

class EchoState:
    def __init__(self, N_G=N_G, g_GG=g_GG, g_Gz=g_Gz, 
        p_GG=p_GG, p_z=p_z, dt=dt, target_f = target_f, I = I, 
        Δt=Δt, α=α, τ=τ, N_output=N_output):

        self.N_G, self.g_GG, self.g_Gz, self.α = N_G, g_GG, g_Gz, α
        self.p_GG, self.p_z = p_GG, p_z
        self.Δ, self.dt, self.τ = Δ, dt, τ
        self.N_output = N_output
        self.f = target_f
        self.I = I

        self.conditioned_neurons = 10

        self.J_GG = sparse.random(self.N_G, self.N_G, density=self.p_GG,
                data_rvs=np.random.randn).toarray()*1/np.sqrt(self.p_GG*self.N_G)

        self.J_Gz = np.zeros((self.N_G, self.N_output))
        for cneuron in self.conditioned_neurons:
            self.J_Gz[cneuron, :] = 1.
        
        # assert(self.I.shape == (self.))
        self.J_GI = np.tile(sparse.rand(self.N_G, 1, density=1/self.N_G).toarray(), self.N_G)

        print(self.J_GI)

        self.J_GG_initial = deepcopy(self.J_GG)
        self.J_Gz_initial = deepcopy(self.J_Gz)
        
    def _init_variables(self):

        self.nb_train_steps = 0
        self.time_elapsed = 0

        self.w = np.zeros((self.N_G, self.N_output))
        self.x = np.random.randn(self.N_G)
        self.r = np.tanh(self.x)
        self.z = np.random.randn(self.N_output)
        self.P = np.eye(self.N_G)/self.α

        self.w_list = []
        self.dw_list = []

        self.z_list = {}
        self.z_list['train'], self.z_list['test'] = [], []
    
    def step(self, train_test='train', store=True):

        dt, Δt, τ, g_GG, g_Gz = self.dt, self.Δt, self.τ, self.g_GG, self.g_Gz

        self.dx = 1/τ*(-self.x + g_GG*np.dot(self.J_GG, self.r) + \
            g_Gz*np.dot(self.J_Gz, self.z) + np.dot(self.J_GI, self.I[:, int(self.time_elapsed/dt)]))
        
        self.x = self.x + dt*self.dx
        self.r = np.tanh(self.x)
        self.z = np.dot(self.w.T, self.r).flatten()
        self.time_elapsed += dt

        self.dw = np.zeros_like(self.w)

        if train_test == 'train':
            self.nb_train_steps += 1
            if self.nb_train_steps%(Δt//dt) == 0:
                Pr = np.dot(self.P, self.r)

                self.P -= np.outer(Pr, self.r).dot(self.P)/(1+np.dot(self.r, Pr))

                self.e_minus = self.z - self.f(self.time_elapsed)

                self.dw = np.outer(np.dot(P, self.r), self.e_minus)
                self.w -= self.dw
            
            if store:
                self.w_list.append((self.time_elapsed, np.linalg.norm(self.w, axis=0)))
                self.dw_list.append((self.time_elapsed, np.linalg.norm(self.dw, axis=0)))

                self.z_list[train_test].append((self.time_elapsed, self.z))

    



        

