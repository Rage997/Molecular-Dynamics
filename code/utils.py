import numpy as np
from scipy.spatial import KDTree

def LJForce(displacement):
    # TODO check this
    r = np.linalg.norm(displacement, 2)
    force = 48/(r**2)*(1/(r**12)-0.5*1/(r**6))*displacement
    return force

def VelocityVerletTimeStepping(currentBox):
    '''Integrate in space using Verlet scheme'''
    previousParticleForces = currentBox.particleForces
    currentBox.particlePositions = (currentBox.particlePositions + currentBox.particleVelocities*currentBox.dt + 0.5*currentBox.particleForces*(currentBox.dt)**2)%(currentBox.boxLength)
    currentBox = currentBox.evaluateForce()
    currentBox.particleVelocities = currentBox.particleVelocities + 0.5*(previousParticleForces + currentBox.particleForces)*currentBox.dt
    currentBox.tree = KDTree(currentBox.particlePositions, leafsize=2000) # update
    
    return currentBox