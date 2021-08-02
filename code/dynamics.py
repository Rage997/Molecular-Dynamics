import itertools
import numpy as np
import scipy
from scipy import constants
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
        self.particlePositionsPrev = np.zeros((numberOfParticles, dimension))
        self.particleVelocities = (np.random.rand(numberOfParticles, dimension)-0.5) #assign randomly
        self.particleForces = np.zeros((numberOfParticles, dimension))
        self.potential_energy = 0
        self.etot = 0 # TODO refactor
        self.latticePositions()

        # self.updateVerletList()

    def updateVerletList(self):
        # To obtain O(n) particle interaction computation
        sz = int(np.ceil(self.boxLength/self.rc))
        self.verletList = np.empty((sz, sz), object)
        # self.verletList.fill([])
        for i in np.ndindex(self.verletList.shape):
            self.verletList[i] = []
        for i in range(self.numberOfParticles):
            y_idx = int(np.ceil(self.particlePositions[i, 0] / self.rc))
            x_idx = int(np.ceil(self.particlePositions[i, 1] / self.rc))
            if len(self.verletList[y_idx, x_idx]) == 0:
                self.verletList[y_idx, x_idx].append(i)
            else:
                self.verletList[y_idx, x_idx] = [i]

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

        self.particlePositionsPrev = self.particlePositions - self.particleVelocities * self.dt

    def evaluateKineticEnergy(self):
        return 0.5*np.sum(np.square(self.particleVelocities))

    def evaluatePotentialEnergy(self):
        return self.potential_energy

    def evaluateTotalEnergy(self):
        return self.evaluateKineticEnergy() + self.evaluatePotentialEnergy()
        # return self.etot

    def evaluateForce(self):
        #force = - gradient of potential
        self.particleForces = np.zeros((self.numberOfParticles, self.dimension))
        self.potential_energy = 0 # reset the potential energy to zero. It will be updated during force computation
        for i in range(self.numberOfParticles):
            # Get neighbourhoods from verlet list
            # y_idx = int(np.ceil(self.particlePositions[i, 0] / self.rc)) -1
            # x_idx = int(np.ceil(self.particlePositions[i, 1] / self.rc)) -1
            # cell = self.verletList[y_idx, x_idx]
            # # Get all the neighbourhoods
            # sz = self.verletList.shape[0]
            # top = self.verletList[(y_idx+1)%sz, x_idx]
            # top_right = self.verletList[(y_idx+1)%sz, (x_idx+1)%sz]
            # top_left = self.verletList[(y_idx+1)%sz, (x_idx-1)%sz]
            # right = self.verletList[y_idx, (x_idx+1)%sz]
            # left = self.verletList[y_idx, (x_idx-1)%sz]
            # bottom = self.verletList[(y_idx-1)%sz, x_idx]
            # bottom_right = self.verletList[(y_idx-1)%sz, (x_idx+1)%sz]
            # bottom_left = self.verletList[(y_idx-1)%sz, (x_idx-1)%sz]

            # neighbourhood = np.concatenate([cell, top, top_right, top_left, right, left,
            #                                 bottom, bottom_left, bottom_right]).astype(int)

            for j in range(i+1, self.numberOfParticles): # if you don't want to use verlet list
            # for j in neighbourhood:
            #     if i != j:
                xr = self.particlePositions[i,:]-self.particlePositions[j,:]
                xr -= self.boxLength*np.ceil(xr/self.boxLength)
                self.particleForces[i,:] += self.LJForce(xr)
                self.particleForces[j,:] += -self.particleForces[i,:]

    def LJPotential(self, r):
        return 4*self.epsilon*np.power(self.sigma/r, 12)/r - np.power(self.sigma/r, 6) 

    def LJForce(self, xr):

        rr = np.linalg.norm(xr)
        # r2 = np.sqrt(rr)
        # r2 = rr
        ecut = self.LJPotential(self.rc)
        if rr > self.rc:
            force = 0
        else:
            r2i = self.sigma / rr                
            r6i = np.power(r2i, 3)
            force = 48*r2i *r6i* (r6i - 0.5)
            force *= xr
            # Update energy (which one is correct??)
            # self.potential_energy += self.LJPotential(rr) - ecut 
            self.potential_energy += 4*r6i*(r6i-1) - ecut 
        return force

    def IntegrateVerlet(self):
        '''Integrate in space using Verlet scheme'''
        
        sumv = 0
        sumv2 = 0
        for i in range(self.numberOfParticles):
            xx = 2*self.particlePositions[i,:] - self.particlePositionsPrev[i,:] + self.dt**2*self.particleForces[i]
            vi = (xx-self.particlePositionsPrev[i,:])/(2*self.dt)
            sumv += vi
            sumv2 += vi**2
            self.particlePositionsPrev[i,:] = self.particlePositions[i, :]
            self.particlePositions[i, :] = xx

            # Update verlet list
            # y_idx = int(np.ceil(self.particlePositions[i, 0] / self.rc))
            # x_idx = int(np.ceil(self.particlePositions[i, 1] / self.rc))
            # if len(self.verletList[y_idx, x_idx]) == 0:
            #     self.verletList[y_idx, x_idx].append(i)
            # else:
            #     self.verletList[y_idx, x_idx] = [i]

        temp = sumv2/(3*self.numberOfParticles)
        self.etot = (self.potential_energy + 0.5*sumv2)/self.numberOfParticles
        # print(self.potential_energy)
        # print(self.etot)

        # self.particlePositions = self.particlePositions + self.particleVelocities*self.dt + 0.5*self.particleForces*(self.dt)**2
        # # periodicity
        # self.particlePositions = np.mod(self.particlePositions, [self.boxLength, self.boxLength])
        # # Updating the verlet list takes O(n) time ...
        # self.updateVerletList()

    def evaluateTotalMomentum(self):
        # the mass is 1 for each particle
        return np.sum(self.particleVelocities)

    def thermostat(self, T_bath):
        '''Berendsen thermostat'''
        dt_inv_tau = 0.025 
        T = 2/3 *(self.evaluateKineticEnergy()) / (self.numberOfParticles * constants.Boltzmann)
        gamma = np.sqrt(1 + dt_inv_tau (T_bath/ T - 1))
        self.particleVelocities *= gamma

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
        plt.close()