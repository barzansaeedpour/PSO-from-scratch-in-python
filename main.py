

import numpy as np
import math
from problem import problem
import copy
import matplotlib.pyplot as plt

##################### problem definition

def costFunction(x):
    return problem(x)

nVar = 10              # Number of Decision Variables
varSize = nVar        # Size of Decision Variable Matrix
varMin = -20          # Lower Bound of Variables
varMax = 20           # Upper Bound of Variables

##################### PSO parameters

maxIt = 100           # Maximun number of iterations
nPop = 50             # Population size (Swarm size)

constiction_Coefficient = True

if constiction_Coefficient == False:
    w = 1                 # Inertia Weight
    w_damp = 0.99         # Inertia Weight Damping Ratio
    c1 = 2                # Personal learning Coefficient
    c2 = 2                # Global learning Coefficient
else:
    ##################### Constriction Coefficients
    phi1 = 2.05
    phi2 = 2.05
    phi = phi1 + phi2
    chi = 2 / (phi - 2 + np.sqrt((phi**2) - (4 * phi)))
    w = chi               
    w_damp = 1         
    c1 = chi * phi1                
    c2 = chi * phi2     

##################### Initialization

class Particle:
    pass

empty_particle = Particle()

empty_particle.position = np.array([])
empty_particle.cost = None
empty_particle.velocity = np.array([])
empty_particle.best_position = np.array([])
empty_particle.best_cost = None

pop = np.array([])              # The population

# Create a population of empty particles:
for i in range(nPop):
    pop = np.append(pop, copy.copy(empty_particle))


global_best_particle = copy.copy(empty_particle)
global_best_particle.cost = math.inf


# Initialize the particles
for particle in pop:
    particle.position = np.random.uniform(low=varMin, high=varMax, size=varSize)
    particle.velocity = np.zeros(shape=varSize)
    particle.cost = costFunction(particle.position)
    particle.best_position = particle.position
    particle.best_cost = particle.cost

    if particle.best_cost < global_best_particle.cost:
        global_best_particle = copy.copy(particle)


best_cost_list = []     # we will use this list to plot the best costs at the end
best_cost_list.append(global_best_particle.cost)


# PSO Main Loop

for itr in range(maxIt):
    for i, particle in enumerate(pop):
        
        # update velocity
        # r1 = np.random.random()
        # r2 = np.random.random()
        r1 = np.random.random(size=varSize)
        r2 = np.random.random(size=varSize)
        particle.velocity = (w * particle.velocity) + (c1 * r1 * (particle.best_position - particle.position)) + (c2 * r2 * (global_best_particle.position - particle.position))

        # update position
        particle.position = particle.position + particle.velocity 
        
        # update cost
        particle.cost = costFunction(particle.position)
        
        # update personal Best
        if particle.cost < particle.best_cost:
            particle.best_cost = particle.cost
            particle.best_position = particle.position
            
            # update global Best
            if particle.cost < global_best_particle.cost:
                global_best_particle = copy.copy(particle)

    best_cost_list.append(global_best_particle.cost)
    print(global_best_particle.cost)
    w = w * w_damp

plt.plot(best_cost_list)
plt.title('Cost')
plt.xlabel('Iterations')
plt.ylabel('Cost (Linear)')

plt.figure()
plt.semilogy(best_cost_list)
plt.title('Cost')
plt.xlabel('Iterations')
plt.ylabel('Cost (Semilog)')
plt.show()

print('Finished')












