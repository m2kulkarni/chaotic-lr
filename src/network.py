import numpy as np
import matplotlib.pyplot as plt

class EchoState:
    def __init__(self, N_G=N_G, g_GG=g_GG, g_Gz=g_Gz, 
        p_GG=p_GG, p_z=p_z, dt=dt, target_f = target_f, 
        Δt=Δt, α=α, τ=τ, N_output=N_output):

        self.N_G, self.g_GG, self.g_Gz, self.α = N_G, g_GG, g_Gz, α
        self.p_GG, self.p_z = p_GG, p_z
        self.Δ, self.dt, self.τ = Δ, dt, τ

        self.conditioned_neurons = 10

        




        

