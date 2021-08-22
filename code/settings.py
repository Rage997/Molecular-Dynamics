'''Settings for molecular dynamics simulation'''

numberOfParticles = 49
dimension = 2 # this should work also in 3D but haven't tested much. Future work
# Truncated lennard-jones potential params
sigma = 1
epsilon = 1
rc = 2.5
temperature = 0.5
boxLength = 20 # parameter L in assignment
dt = 0.0001 #time step
T_max = 1