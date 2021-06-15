import itertools
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

random.seed(time.time())
class MolecularDynamics:
    def __init__(self, numberOfParticles, boxLength, dimension, sigma, epsilon,
                rc, temperature, dt):
        self.numberOfParticles = numberOfParticles
        self.boxLength = boxLength
        self.dimension = dimension
        self.sigma = sigma
        self.epsilon = epsilon
        self.rc = rc
        self.temperature = temperature
        self.dt = dt #time step 
        self.nrho = numberOfParticles/(boxLength**(dimension)) # particles density

        # Init particles positions and velocities
        self.particlePositions = np.zeros((numberOfParticles, dimension))
        self.particleVelocities = (np.random.rand(numberOfParticles, dimension)-0.5) #assign randomly
        self.particleForces = np.zeros((numberOfParticles, dimension))
        self.energy = 0
        self.latticePositions()

    def latticePositions(self):
        '''Assigns particles randomly in a regular lattice'''

        pointsInLattice = math.ceil(self.numberOfParticles**(1/self.dimension))
        spots = np.linspace(1, self.boxLength, num=pointsInLattice, endpoint=False)
        count = 0
        for p in itertools.product(spots, repeat=self.dimension):
            # we "insert" each line sequentially into the regular lattice
            p = np.asarray(list(p))
            self.particlePositions[count, :] = p
            count += 1
            if count>self.numberOfParticles-1:
                break
        # Scale factor of velocities
        sumv2 = np.mean(self.particleVelocities**2, axis=0)
        fs = np.sqrt(3*self.temperature/sumv2)
        # Ensure that momentum is zero
        v_cm = np.mean(self.particleVelocities, axis=0)
        self.particleVelocities = fs*(self.particleVelocities - v_cm)

    def evaluateTotalEnergy(self):
        # Kinetic energy is not required apparently
        # K = 0.5*np.sum(np.square(self.particleVelocities)) 
        # return self.PotentialEnergy + K
        return self.energy

    def evaluateForce(self):
        #force = - gradient of potential
        
        self.particleForces = np.zeros((self.numberOfParticles, self.dimension))
        for i in range(self.numberOfParticles-1):
            for j in range(i+1, self.numberOfParticles):
                xr = self.particlePositions[i,:]-self.particlePositions[j,:]
                xr -= self.boxLength*np.ceil(xr/self.boxLength)
                self.particleForces[i,:] += self.LJForce(xr)
                self.particleForces[j,:] += -self.particleForces[i,:]

    def LJForce(self, xr):
        r = np.linalg.norm(xr, 2)
        ecut = (4*self.epsilon*((self.sigma/self.rc)**12-(self.sigma/self.rc)**6))
        if r > self.rc:
            force = 0
        else:
            force = 48/(r**2)*(1/(r**12)-0.5*1/(r**6))*xr - ecut
        # Update energy
        self.energy += 4*(1/r**6)*(1/r**6 - 1) - ecut
        return force

    def IntegrateVerlet(self):
        '''Integrate in space using Verlet scheme'''
        
        self.particlePositions = self.particlePositions + self.particleVelocities*self.dt + 0.5*self.particleForces*(self.dt)**2
        # periodicity
        self.particlePositions = np.mod(self.particlePositions, [self.boxLength, self.boxLength])

    def evaluateTotalMomentum(self):
        # the mass is 1 for each particle
        return np.sum(self.particleVelocities)

    def output(self, filename):
        '''This can be used for 3D visualization'''
        file_xyz = open(filename, 'w')
        file_xyz.write(str(self.numberOfParticles) +'\n')
        # file_xyz.write('Step: %s	Time: %sps	Temperature: %sK\n' % (step, Time, temp))
        for i in range(self.numberOfParticles):
            file_xyz.write("{} {} \n".format(self.particlePositions[i][0], self.particlePositions[i][1]))

    def plot(self, filename, timestep):
        '''For 2D, I suggest to use the good and old matplotlib'''
        plt.figure()
        plt.title("System at timestep {}".format(timestep))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-0.1*self.boxLength, 1.05*self.boxLength])
        plt.ylim([-0.1*self.boxLength, 1.05*self.boxLength])
        plt.plot(self.particlePositions[:, 0], self.particlePositions[:, 1], '.r')
        plt.savefig(filename)
