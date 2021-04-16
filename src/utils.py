import numpy as np
import matplotlib.pyplot as plt


def periodic(t, amp=3., freq=1/300):
    """Generates a periodic function which a sum of 4 sinusoids.
    """
    return amp*np.sin(np.pi*freq*t) + (amp/2) * np.sin(2*np.pi*freq*t) + (amp/3) * np.sin(3*np.pi*freq*t) + (amp/4) * np.sin(4*np.pi*freq*t)
periodic = np.vectorize(periodic)

def clamped(t, clamp=1.):
    # print(np.int(t))
    if t > 200 and t<250:
        return clamp*1.5
    if t> 500+200 and t<500+250:
        return clamp*1.2
    if t>1000+200 and t<1000+250:
        return clamp*0.7
    else:
        return clamp
clamp = np.vectorize(clamped)

def triangle(t, freq=1/600, amp=3):
    """Generates a triangle-wave function.
    """
    return amp*signal.sawtooth(2*np.pi*freq*t, 0.5)
triangle = np.vectorize(triangle)

def cos_fun(t, amp=3., freq=1/300):
    """Generates a cos function.
    """
    return amp*np.cos(np.pi*freq*t)
cos_fun = np.vectorize(cos_fun)

def complicated_periodic(t, amp=1., freq=1/300, seed=1):
    """Generates a complicated periodic function which a sum of 10 sinusoids.
    """
    np.random.seed(seed)
    amps = np.random.randint(1, 5, size=(6,))
    freqs = np.random.randint(1, 10, size=(6,))
    return sum(am*amp*np.sin(fr*np.pi*freq*t) for am, fr in zip(amps, freqs))
complicated_periodic = np.vectorize(complicated_periodic)

def both(f, g):
    """Generates  the function \\\(t ⟼ (f(t), g(t))\\\)
    """
    return (lambda t: np.array([f(t), g(t)]) if isinstance(t, float) else np.array(list(zip(f(t), g(t)))))
per_tri = both(periodic, triangle)

def triple(f, g, h):
    """Generates  the function \\\(t ⟼ (f(t), g(t), h(t))\\\)
    """
    return (lambda t: np.array([f(t), g(t), h(t)]) if isinstance(t, float) else np.array(list(zip(f(t), g(t), h(t)))))
per_tri_cos = triple(periodic, triangle, cos_fun)

N_G = 1000
g_GG = 1.5
g_Gz = 1.
p_GG = 0.1 
p_z = 1.
dt = 0.05
Δt =  0.5
α = 1.
τ = 5.
N_output = 1
target_f = clamp
I = 0

