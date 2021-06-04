import dynamics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# Parameters of simulation
numberOfParticles = 100
dimension = 2 # this should work also in 3D but haven't tested much. Future work
# Truncated lennard-jones potential params
sigma = 1
epsilon = 1
rc = 2.5

temperature = 1
L = 30
dt = 0.0001 #time step
T_max = 1000

#set up the simulation
simulation = dynamics.MolecularDynamics(numberOfParticles, L, dimension, 
                        sigma, epsilon, rc, temperature, dt)

#calculating forces on particles in the box
simulation.evaluateForce()

#making a list of particle positions and velocities for acf and what not. You could make an animation out of this
particlePositionsList = [simulation.particlePositions]
particleVelocityList = [simulation.particleVelocities]

#making a list of energies at various time steps to plot later
energy = np.zeros(T_max+1,)
energy[0] = simulation.evaluateTotalEnergy()
momentum = np.zeros(T_max+1,)
momentum[0] = simulation.evaluateTotalMomentum()


for i in range(T_max):
    # simulation.plot('system{}.png'.format(i))

    #do the time step, knowing that simulation already knows the particle forces at the moment
    simulation.IntegrateVerlet() #evaluates forces on particles, updates particle positions and velocities
    energy[i+1] = simulation.evaluateTotalEnergy()
    momentum[i+1] = simulation.evaluateTotalMomentum()

    particlePositionsList.append(simulation.particlePositions)
    particleVelocityList.append(simulation.particleVelocities)

# Plotting
timepoints = np.arange(T_max+1)*simulation.dt
fig, ax = plt.subplots(2)

ax[0].plot(timepoints, energy)
ax[0].set_title("Energy over time. The fluctuation are in order sqrt(n).")
ax[0].set_xlabel("time")
ax[0].set_ylabel("energy")
ax[0].set_ylim(np.amin(energy)*0.999, np.amax(energy)*1.001)
# plt.show()
# plt.savefig('./energy.png')

ax[1].plot(timepoints, momentum)
ax[1].set_title("Momentum over time. It shoult close to zero.")
ax[1].set_xlabel("time")
ax[1].set_ylabel("momentum")
ax[1].set_ylim(np.amin(momentum)*0.999, np.amax(momentum)*1.001)
# plt.show()
plt.savefig('./plot.png')
