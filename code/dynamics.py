import itertools
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from utils import LJForce, VelocityVerletTimeStepping

random.seed(time.time())

#create a box of particles
#make object Box which will hold all the particles
class Box:
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

        self.particlePositions = np.zeros((numberOfParticles, dimension))
        self.particleVelocities = self.boxLength*(np.random.rand(numberOfParticles, dimension)-0.5) #assign randomly
        self.particleForces = np.zeros((numberOfParticles, dimension))

    def latticePositions(self):
        '''Assigns particles randomly in a regular lattice'''

        pointsInLattice = math.ceil(self.numberOfParticles**(1/self.dimension))

        spots = np.linspace(0, self.boxLength, num=pointsInLattice, endpoint=False)
        count = 0
        for p in itertools.product(spots, repeat=self.dimension):
            # we "insert" each line sequentially into the regular lattice
            p = np.asarray(list(p))
            self.particlePositions[count, :] = p
            count += 1
            if count>self.numberOfParticles-1:
                break
        return self

    def evaluateKineticEnergy(self):
        return 0.5*np.sum(np.square(self.particleVelocities))

    def evaluatePotentialEnergy(self):
        energy = 0
        for i in range(self.numberOfParticles):
            for j in range(i+1, self.numberOfParticles):
                displacement = self.particlePositions[i,:]-self.particlePositions[j,:]
                for k in range(self.dimension):
                    # this ensures the periodicity of the box
                    if abs(displacement[k])>self.boxLength/2:
                        displacement[k] -= self.boxLength*np.sign(displacement[k])
                r = np.linalg.norm(displacement,2)
                # Using this makes energy fluctutate. 
                if r > self.rc:
                    energy += 0 # truncate potential
                else:    
                    energy += (4*self.epsilon*((self.sigma/r)**12-(self.sigma/r)**6))
        return energy

    def evaluateTotalEnergy(self):
        return self.evaluatePotentialEnergy()+self.evaluateKineticEnergy()

    def evaluateForce(self):
        #force = - gradient of potential
        self.particleForces = np.zeros((self.numberOfParticles, self.dimension))
        for i in range(self.numberOfParticles):
            for j in range(i+1, self.numberOfParticles):
                rij = self.particlePositions[i,:]-self.particlePositions[j,:]
                for k in range(self.dimension):
                    # this ensures the periodicity of the box
                    if abs(rij[k])>self.boxLength/2:
                        rij[k] -= self.boxLength*np.sign(rij[k])
                rji = -rij
                self.particleForces[i,:] += LJForce(rij)
                self.particleForces[j,:] += -self.particleForces[i,:]
        return self

    # todo this can be put in init
    def stationaryCenterOfMass(self):
        '''Ensures that total momentum is zero'''
        v_cm = np.mean(self.particleVelocities, axis=0)
        self.particleVelocities = self.particleVelocities - v_cm
        return self

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

    def plot(self, filename):
        '''For 2D, I suggest to use the good and old matplotlib'''
        plt.title("System at timestep ...")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(self.particlePositions[:, 0], self.particlePositions[:, 1], '.r')

        plt.savefig(filename)



#   self.initVerletList()
        
#     def initVerletList(self):
#         size = int(np.ceil(self.boxLength/self.rc))
#         self.verletList =  np.empty( (size,size), dtype=object)
#         for i in range(self.numberOfParticles):
#             pos = self.particlePositions[i,:]
#             idx = np.ceil(pos)
#             ii, jj = int(idx[0]), int(idx[1])
#             if self.verletList[ii, jj]:
#                 self.verletList[ii, jj].append(i)
#             else:
#                 self.verletList[ii, jj] = [i]