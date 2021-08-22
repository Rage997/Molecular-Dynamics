import moleculary_dynamics
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# Parameters of simulation
numberOfParticles = 500
dimension = 2 # this should work also in 3D but haven't tested much. Future work
# Truncated lennard-jones potential params
sigma = 1
epsilon = 1
rc = 2.5
temperature = 0.1
L = 30
dt = 1e-6 #time step
T_max = 1

#set up the simulation
simulation = moleculary_dynamics.MolecularDynamics(numberOfParticles, L, dimension, 
                        sigma, epsilon, rc, temperature, T_max, dt)
simulation.run()

simulation.plot()