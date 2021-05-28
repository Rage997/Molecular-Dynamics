import dynamics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# Parameters of simulation
numberOfParticles = 1000
dimension = 2 # this should work also in 3D but haven't tested much. Future work
# Truncated lennard-jones potential params
sigma = 1
epsilon = 1
rc = 2.5
temperature = 1

L = 30
dt = 0.001 #time step
T_max = 1000

#set up the box
currentBox = dynamics.Box(numberOfParticles, L, dimension, 
                        sigma, epsilon, rc, temperature, dt)

#placing particles in a lattice
currentBox = currentBox.latticePositions()

#ensuring box has no net momentum
currentBox = currentBox.stationaryCenterOfMass()

#calculating forces on particles in the box
currentBox = currentBox.evaluateForce()

#making a list of particle positions and velocities for acf and what not. You could make an animation out of this
particlePositionsList = [currentBox.particlePositions]
particleVelocityList = [currentBox.particleVelocities]

#making a list of energies at various time steps to plot later
energy = np.zeros(T_max+1,)
energy[0] = currentBox.evaluateTotalEnergy()
momentum = np.zeros(T_max+1,)
momentum[0] = currentBox.evaluateTotalMomentum()


for i in range(T_max):
    # currentBox.plot('system{}.png'.format(i))

    #do the time step, knowing that currentBox already knows the particle forces at the moment
    currentBox = dynamics.VelocityVerletTimeStepping(currentBox) #evaluates forces on particles, updates particle positions and velocities
    energy[i+1] = currentBox.evaluateTotalEnergy()
    momentum[i+1] = currentBox.evaluateTotalMomentum()

    particlePositionsList.append(currentBox.particlePositions)
    particleVelocityList.append(currentBox.particleVelocities)

# ACF computation
ACF = np.zeros(T_max+1,)
for i in range(T_max+1):
    for j in range(T_max+1-i):
        ACF[i] = ACF[i] + np.sum(particleVelocityList[j]*particleVelocityList[j+i])
        # ACF[j] = ACF[j] + np.sum(particleVelocityList[i]*particleVelocityList[j+i]) #which one is correct??
    ACF[i] = ACF[i]/(T_max+1-i)


# Plotting
timepoints = np.arange(T_max+1)*currentBox.dt
fig, ax = plt.subplots(3)

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
# plt.savefig('./energy.png')

ax[2].plot(timepoints, ACF)
ax[2].set_title("Normalized VACF plot")
ax[2].set_xlabel("time")
ax[2].set_ylabel("VACF")
# plt.show()
plt.savefig('./vacf.png')
