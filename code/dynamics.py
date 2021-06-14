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
                    # periodicity of the box
                    if abs(rij[k])>self.boxLength/2:
                        rij[k] -= self.boxLength*np.sign(rij[k])
                # rji = -rij
                self.particleForces[i,:] += self.LJForce(rij)
                self.particleForces[j,:] += -self.particleForces[i,:]

    def LJForce(self, xr):
        r = np.linalg.norm(xr, 2)
        if r > self.rc:
            force = 0
        else:
            force = 48/(r**2)*(1/(r**12)-0.5*1/(r**6))*xr
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

    def plot(self, filename):
        '''For 2D, I suggest to use the good and old matplotlib'''
        plt.figure()
        plt.title("System at timestep ...")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-0.1*self.boxLength, 1.05*self.boxLength])
        plt.ylim([-0.1*self.boxLength, 1.05*self.boxLength])
        plt.plot(self.particlePositions[:, 0], self.particlePositions[:, 1], '.r')
        plt.savefig(filename)
