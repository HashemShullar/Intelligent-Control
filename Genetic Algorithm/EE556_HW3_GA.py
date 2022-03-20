import numpy as np
import copy
import cmath



Ao = np.array([[0, 377, 0, 0] , [-0.0587, 0, -0.1303, 0], [-0.0899, 0, -0.1956, 0.1289], [95.605, 0, -816.0862, -20]])


EigenOpen = np.linalg.eig(Ao)[0]


K  = 7.09;
T1 = 0.6851;
T2 = 0.1;
Ac = np.array([[0, 377, 0, 0, 0, 0],
               [-0.0587, 0, -0.1303, 0, 0, 0],
               [-0.0899, 0, -0.1956, 0.1289, 0, 0],
               [95.605, 0, -816.0862, -20, 0, 1000],
               [-0.0587, 0, -0.1303, 0, -0.333, 0],
               [-0.0587*(K*T1/T2), 0, -0.1303*(K*T1/T2), 0, K/T2*(1-(T1/3)), -1/T2]])

EigenClosed = np.linalg.eig(Ac)[0]
print(EigenClosed)

sig = EigenClosed.real
w   = EigenClosed.imag



z = -(sig / (np.sqrt(sig ** 2 + w ** 2)))
print('Minimum Z of Closed Loop Eigenvalue = ', np.min(z))

# Initializing the population:

population_size = 100
crossover_p = 0.8
mutation_p  = 0.01

eigen = np.zeros((6, population_size), dtype = 'complex_') # 6 because we have 6 eigen values
sig   = np.zeros((6, population_size))
w     = np.zeros((6, population_size))

z = np.zeros((6, population_size))
z_min = np.zeros((population_size, 1))
solution = np.zeros((population_size, 4))



for i in range(population_size):


    K  = np.random.uniform(1, 30, 1)[0]
    T1 = np.random.uniform(0.1, 0.5, 1)[0]
    T2 = 0.1 # np.random.uniform(0.01, 0.1, 1)[0]

    Ac = np.array([[0, 377, 0, 0, 0, 0],
                   [-0.0587, 0, -0.1303, 0, 0, 0],
                   [-0.0899, 0, -0.1956, 0.1289, 0, 0],
                   [95.605, 0, -816.0862, -20, 0, 1000],
                   [-0.0587, 0, -0.1303, 0, -0.333, 0],
                   [-0.0587*(K*T1/T2), 0, -0.1303*(K*T1/T2), 0, K/T2*(1-T1/3), -1/T2]])




    eigen[:, i] = np.linalg.eig(Ac)[0]
    tempr = copy.deepcopy(eigen[:, i])
    tempi = copy.deepcopy(eigen[:, i])
    sig[:, i]   = eigen[:, i].real
    w[:, i]     = eigen[:, i].imag
    NumOfEig    = 6

    for j in range(NumOfEig):
        z[j, i] = -(sig[j, i] / (np.sqrt(sig[j, i] ** 2 + w[j, i] ** 2)) )


    z_min[i,:] = np.min(z[:, i])
    solution[i,:] = [K, T1, T2, z_min[i][0]]



# Selection

new_population = np.zeros((population_size, 4))
tournament = np.zeros((population_size, 2))



for i in range(population_size):
    select1, select2 = np.random.choice(population_size, size=(2, 1), replace=False)
    tournament[i, :] = select1[0], select2[0]



    if np.random.uniform(0, 1, 1)[0] > 0.2:
        if solution[int(tournament[i, 0]), 3] >= solution[int(tournament[i, 1]), 3]:
            new_population[i,:] = solution[int(tournament[i, 0]),:]
        else:
            new_population[i,:] = solution[int(tournament[i, 1]),:]

    else:
        if solution[int(tournament[i, 0]), 3] <= solution[int(tournament[i, 1]), 3]:
            new_population[i,:] = solution[int(tournament[i, 0]),:]
        else:
            new_population[i,:] = solution[int(tournament[i, 1]),:]


# zz = [1, 2, 3, 4, 5]
# print(zz[0:3]) # printed [1, 2, 3]

# Flat crossover

offspring = np.zeros((population_size, 3))

for i in range(population_size-1):
    if np.random.uniform(0, 1, 1)[0] < crossover_p:
        for j in range(3):
            if new_population[i, j] >= new_population[i+1, j]:
                offspring[i, j] = np.random.uniform(new_population[i+1, j], new_population[i, j], 1)[0]
            else:
                offspring[i, j] = np.random.uniform(new_population[i, j], new_population[i+1, j], 1)[0]

    else:
        offspring[i, 0:3] = new_population[i, 0:3];

offspring[population_size-1, 0:3] = new_population[population_size-1, 0:3];



# Mutation

mutated_offspring = copy.deepcopy(offspring)


for i in range(population_size):

    if np.random.uniform(0, 1, 1)[0]  <= mutation_p:
        id = np.random.choice(3, 1, replace=False)
        if id == 1:
            mutated_offspring[i, id] = np.random.uniform(1, 100, 1)[0]

        if id == 2:
            mutated_offspring[i, id] = np.random.uniform(0.1, 1, 1)[0]

        if id == 3:
            mutated_offspring[i, id] = np.random.uniform(0.01, 0.1, 1)[0]

    else:
        mutated_offspring[i,:] = offspring[i,:]


# Choosing Optimal Solution:


K  = mutated_offspring[:, 0];
T1 = mutated_offspring[:, 1];
T2 = mutated_offspring[:, 2];

for i in range(population_size):

    Ac = np.array([[0, 377, 0, 0, 0, 0],
                   [-0.0587, 0, -0.1303, 0, 0, 0],
                   [-0.0899, 0, -0.1956, 0.1289, 0, 0],
                   [95.605, 0, -816.0862, -20, 0, 1000],
                   [-0.0587, 0, -0.1303, 0, -0.333, 0],
                   [-0.0587*(K[i]*T1[i]/T2[i]), 0, -0.1303*(K[i]*T1[i]/T2[i]), 0, K[i]/T2[i]*(1-T1[i]/3), -1/T2[i]]])


    eigen[:, i] = np.linalg.eig(Ac)[0]
    tempr = copy.deepcopy(eigen[:, i])
    tempi = copy.deepcopy(eigen[:, i])
    sig[:, i]   = eigen[:, i].real
    w[:, i]     = eigen[:, i].imag


    for j in range(NumOfEig):
        z[j, i] = -(sig[j, i] / (np.sqrt(sig[j, i] ** 2 + w[j, i] ** 2)) )


    z_min[i,:] = np.min(z[:, i])
    solution[i,:] = [K[i], T1[i], T2[i], z_min[i][0]]


OptimalSolution = solution[np.argmax(solution[:, 3]), 0:3]
print('Optimal Solution = ', OptimalSolution, '\nGiving a Z value of ', solution[np.argmax(solution[:, 3]), 3])
print('Population Size =', population_size,'\nCrossover probability =', crossover_p, '\nMutation Rate =', mutation_p )


population_size = 200
crossover_p = 0.8
mutation_p  = 0.1












