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
    def __init__(self, nTotal, boxLength, dimension, sigma, epsilon,
                rc, temperature, T_max, dt):
        self.nTotal = nTotal
        # assert(self.nTotal % 2 == 0)

        self.boxLength = boxLength
        self.dimension = dimension
        self.sigma = sigma
        self.epsilon = epsilon
        self.rc = rc
        self.phicutoff = 4*(self.rc**12 - self.rc**-6)
        self.temperature = temperature
        self.dt = dt #time step 
        self.nrho = nTotal/(boxLength**(dimension)) # particles density
        self.T_max = T_max
        # Init particles positions and velocities
        self.pos = np.zeros((nTotal, dimension))
        # self.posPrev = np.zeros((nTotal, dimension))
        self.velocity = (np.random.rand(nTotal, dimension)-0.5) #assign randomly
        self.force = np.zeros((nTotal, dimension))
        
        self.momentum = []
        # self.ene_pot = np.zeros(self.nTotal)
        self.ene_pot = 0
        self.ene_kin = np.zeros(self.nTotal)
        self.ene_tot = []
        
    def init_position(self):
        '''Assigns particles randomly in a regular lattice'''
        
        pointsInLattice = math.ceil(self.nTotal**(1/self.dimension))
        particles_x = particles_y = pointsInLattice
        dx = self.boxLength / particles_x
        dy = self.boxLength / particles_y
        n = 0
        for j in range(pointsInLattice):
        # while j < pointsInLattice:
            for i in range(pointsInLattice):
                # for j in range(particles_y):
                pos_x = i*dx + 1
                pos_y = j*dx + 1
                self.pos[n, :] = np.array([pos_x, pos_y])
                if n >= self.nTotal-1:
                    break
                n += 1
      
        sumv = np.mean(self.velocity, axis=0)
        sumv2 = np.mean(self.velocity**2, axis=0)
        fs = np.sqrt(3*self.temperature/sumv2) # Scale factor of velocities
        self.velocity = fs*(self.velocity - sumv) #ensure momentum is zero
        # self.posPrev = self.pos - self.velocity * self.dt

    def evaluateForce(self):
        #initialises and resets force, potential and acceleration 
        r = np.zeros(self.dimension)
        force = np.zeros(self.dimension)
        # self.ene_pot = np.zeros(self.nTotal)
        self.ene_pot = 0
        # acc = [np.zeros(self.nTotal),np.zeros(self.nTotal),np.zeros(self.nTotal)]        
        self.force = np.zeros((self.nTotal, self.dimension))

        #looping over every pair of particles
        for i in range(self.nTotal-1):
            for j in range(i+1, self.nTotal):

                r = self.pos[i] - self.pos[j]
                r -= self.boxLength*np.rint(r/self.boxLength) #boundary condition

                r_abs = np.linalg.norm(r)
                
                if r_abs < self.rc:     #checks if the particles are within the cutoff distance

                    #calculated Lennard Jones potential
                    lj_pot =  (4*( (r_abs**-12) - (r_abs**-6) ) ) - self.phicutoff
                    #updates potential energy array
                    # self.ene_pot[i] += lj_pot
                    # self.ene_pot[j] += lj_pot
                    self.ene_pot += lj_pot

                    #F = -div(V)
                    lj_force = 24 * ((2*r_abs**-13)- (r_abs**-7))

                    #Fx = dV/dr * dr/dx
                    #dr/dx = x/r 
                    force = lj_force * (r/r_abs)

                    #self.accelaration array updated using force due to Lennard jones potential
                    #Fij = -Fji
                    # a(t+dt) = f(t) / m where m = 1 in reduced units 
                    self.force[i] += force
                    self.force[j] -= force

    def integrate_position(self):
        # position
        for i in range(self.nTotal):
            # for k in range(DIM):
            # print(vel[i])
            self.pos[i] += self.velocity[i]*self.dt + 0.5 * self.force[i] * self.dt**2
            self.pos = np.mod(self.pos, [self.boxLength, self.boxLength])
        
    def integrate_velocities(self):
        # v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
        for i in range(self.nTotal):
            self.velocity[i] += 0.5 * self.force[i] * self.dt

    def compute_temperature(self):
        sqspeed = np.zeros(self.nTotal)    
        for i in range(self.nTotal):
            sqspeed[i] = np.sum(self.velocity[i]**2)
            self.ene_kin[i] = sqspeed[i]*0.5     #kinetic energy calculated and stored in ene_kin array
        mean_sq_speed = np.mean(sqspeed)    #mean square speed
        mass = 1
        temp = ((mean_sq_speed)/(self.dimension*constants.Boltzmann))* mass      #temperature calculated using equipartition theorem and (sigma/dt*conver)**2 is used to convert reduced unit back to SI

    def compute_momentum(self):
        return np.sum(self.velocity) # the mass is 1 for each particle

    def compute_total_energy(self):
        # ep = 1.653e-21
        ene_kin_tot = np.sum(self.ene_kin) 
        ene_pot_tot = np.sum(self.ene_pot)
        ene_tot = ene_kin_tot + ene_pot_tot

        # print('kinetic energy: potential energy: total energy')
        # print(ene_kin_tot, ene_pot_tot,  ene_tot)
        return ene_tot
        
    def output(self, filename):
        '''This can be used for further visualization'''
        file_xyz = open(filename, 'w')
        file_xyz.write(str(self.nTotal) +'\n')
        # file_xyz.write('Step: %s	Time: %sps	Temperature: %sK\n' % (step, Time, temp))
        for i in range(self.nTotal):
            file_xyz.write("{} {} \n".format(self.pos[i][0], self.pos[i][1]))

    def plot_timestep(self, filename, timestep):
        '''For 2D, I suggest to use the good and old matplotlib'''
        plt.figure()
        plt.title("System at timestep {}".format(timestep))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-0.1*self.boxLength, 1.05*self.boxLength])
        plt.ylim([-0.1*self.boxLength, 1.05*self.boxLength])
        plt.plot(self.pos[:, 0], self.pos[:, 1], '.r')
        plt.savefig(filename)
        plt.close()

    def plot(self):
        timepoints = np.arange(len(self.ene_tot))
        fig, ax = plt.subplots(2)
        
        ax[0].plot(timepoints, self.ene_tot)
        ax[0].set_title("Energy over time. The fluctuation are in order sqrt(n).")
        ax[0].set_xlabel("time")
        ax[0].set_ylabel("energy")
        ax[0].set_ylim(np.amin(self.ene_tot)*0.95, np.amax(self.ene_tot)*1.05)
        # plt.show()
        # plt.savefig('./energy.png')

        ax[1].plot(timepoints, self.momentum)
        ax[1].set_title("Momentum over time. It shoult close to zero.")
        ax[1].set_xlabel("time")
        ax[1].set_ylabel("momentum")
        ax[1].set_ylim(np.amin(self.momentum)*0.95, np.amax(self.momentum)*1.05)
        # plt.show()
        plt.savefig('./plot.png')

    def run(self):

        self.init_position()
        self.evaluateForce()
        self.compute_temperature()
        # self.compute_total_energy()
        # raise ValueError
        t = 0
        self.plot_timestep('system{}.png'.format(t), t)
        
        nSteps = int(self.T_max/self.dt)
        for t in range(nSteps):
            if t % 250 == 0:
                # self.plot_timestep('system{}.png'.format(t), t)
                print('-------- Timestep: {} ---------'.format(t))
                
            self.compute_temperature()
            
            self.integrate_position()
            self.integrate_velocities()
            self.evaluateForce()

            self.integrate_velocities()
            self.compute_total_energy()
            # For plotting
            self.ene_tot.append(self.compute_total_energy())
            self.momentum.append(self.compute_momentum())
            #
            #  t += self.dt
            # step += 1
            
        self.plot() # final plot