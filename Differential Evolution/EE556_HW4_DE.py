import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import cmath


class Differential_Evolution:
    def __init__(self, n, m, k, bounds, M_rate, Crossover_rate, ProblemNumber):
        """Function intializes the Differential Evolution class. This code was developed as part of an assignment for the course EE 556 (Intelligent Control)
            offerd by EE department in KFUPM. You can find the assignement PDF in the same folder this code is in.

        Parameters:
        -----------
        n: int representing the number of candidate solutions

        m: int representing the number of optimized parameters in each solution

        k: int representing the number of generations

        bounds: ndarray of shape (m, 2)
            Each row represents the min and max values the m^th parameter can take. (Problem 1: x1, x2, x3. Problem 2: K, T1, T2)

        M_rate: float, specifies the mutation rate

        Crossover_rate: float, specifies the crossover rate

        ProblemNumber: int, this parameter specifies the problem you want to solve (Problem 1 or Problem 2 from the HW4 PDF). Your choice will only affect the objective function.



        """

        self.n = n
        self.m = m
        self.k = k
        self.G_Counter = 0 # Generation counter
        self.bounds = bounds
        self.best   = [0, 0] # [Solution, Solution value]
        self.current_population = np.zeros((self.n ,self.m))

        self.M_rate = 0.4 # M_rate
        self.Crossover_rate = 0.6 # Crossover_rate
        self.ProblemNumber = ProblemNumber


    def initialization(self):
        """Function to generate the initial population"""

        for sol in range(self.n):
            for param in range(self.m):
                self.current_population[sol, param] = np.random.uniform(bounds[param, 0], bounds[param, 1], 1)[0]
        # print(self.current_population)



    def Objective_Function(self, Population):
        """ Function to evaluate the objective function at  each solution """

        cost = np.zeros((self.n, 1))


        if self.ProblemNumber == 1:

            for i in range(Population.shape[0]):
                cost[i, 0] = Population[i, 0] ** 2 + 2 * (Population[i, 1] ** 2) + 3 * (Population[i, 2] ** 2) + Population[i, 0]*Population[i, 1] + Population[i, 1]*Population[i, 2] -8*Population[i, 0] -16*Population[i, 1] - 32*Population[i, 2] +110
            return cost
        else:

            for i in range(Population.shape[0]):
                eigen = np.zeros((6, 1), dtype='complex_')  # 6 because we have 6 eigen values
                sig = np.zeros((6, 1))
                w = np.zeros((6, 1))

                z = np.zeros((6, 1))
                # z_min = np.zeros((population_size, 1))
                # solution = np.zeros((population_size, 4))

                Ac = np.array([[0, 377, 0, 0, 0, 0],
                               [-0.0587, 0, -0.1303, 0, 0, 0],
                               [-0.0899, 0, -0.1956, 0.1289, 0, 0],
                               [95.605, 0, -816.0862, -20, 0, 1000],
                               [-0.0587, 0, -0.1303, 0, -0.333, 0],
                               [-0.0587 * (Population[i, 0] * Population[i, 1] / Population[i, 2]), 0, -0.1303 * (Population[i, 0] * Population[i, 1] / Population[i, 2]), 0,
                                Population[i, 0] / Population[i, 2] * (1 - Population[i, 1] / 3), -1 / Population[i, 2]]])

                eigen = np.linalg.eig(Ac)[0]
                sig   = eigen.real
                w     = eigen.imag

                for j in range(6):
                    z[j] = -(sig[j] / (np.sqrt(sig[j] ** 2 + w[j] ** 2)))

                cost[i, 0] = np.min(z)

            return -1 * cost




    def Best_Solution(self):
        """ Function to find the best solution """

        cost = self.Objective_Function(self.current_population)
        self.best[0] = self.current_population[np.argmin(cost), :]
        self.best[1] = np.min(cost)


    def Mutation(self):
        """ This function applies Mutation to find Vi using the equation: 𝑉𝑖 = 𝑋𝑟1 + 𝐹 × (𝑋𝑟2 − 𝑋𝑟3) """
        V = np.zeros((self.n ,self.m))

        for i in range(self.current_population.shape[0]):
            idx = np.random.choice(np.arange(0, self.n, 1), size=3, replace = False)

            while np.where(idx == i)[0].shape[0] != 0:
                idx = np.random.choice(np.arange(0, self.n, 1), size=3, replace = False)

            V[i, :] = self.current_population[idx[0], :] + self.M_rate * (self.current_population[idx[1], :] - self.current_population[idx[2], :])
            # print(V[i, :])
            for param in range(self.m):
                if V[i, param] > self.bounds[param, 1]:
                    V[i, param] = self.bounds[param, 1]

                if V[i, param] < self.bounds[param, 0]:
                    V[i, param] = self.bounds[param, 0]


        return V


    def Crossover(self, V):

        U = np.zeros((self.n ,self.m))

        for i in range(U.shape[0]):
            for param in range(U.shape[1]):
                if np.random.uniform(0, 1, 1)[0] <= self.Crossover_rate:
                    U[i, param] = V[i, param]
                else:
                    U[i, param] = self.current_population[i, param]

        return U


    def Selection(self, U):
        """ This function will generate a new population """

        New_Population = np.zeros((self.n ,self.m))
        cost_parent = self.Objective_Function(self.current_population)
        cost_trial  = self.Objective_Function(U)

        for i in range(self.n):
            if cost_trial[i] <= cost_parent[i]:
                New_Population[i, :] = U[i, :]

            else:
                New_Population[i, :] = self.current_population[i, :]

        self.current_population = New_Population


    def Solve(self):


        history = []
        self.initialization()

        for Generation in range(self.k):

            self.Best_Solution()
            history.append(self.best[1])
            if self.G_Counter > 10:
                break
            if Generation > 1:
                if history[Generation] == history[Generation-1]:
                    self.G_Counter += 1
                else:
                    self.G_Counter = 0
                    


            V = self.Mutation()
            U = self.Crossover(V)
            self.Selection(U)


        return self.best[0], self.best[1], history

""" 

EE 556 (Intelligent control) - HW4

Please make sure you provide parameter bounds in the same format described in the Differential_Evolution class. 


"""



Population_size = 100
NumberofParameters = 3
NumberofGenerations = 30
problem_number = 2

# Part 1:

# mutation_rate = 0.4
# crossover_rate = 0.6
# bounds = np.array([[0, 10], [0, 10], [0, 10]])
# DE = Differential_Evolution(Population_size, NumberofParameters, NumberofGenerations, bounds, mutation_rate, crossover_rate, problem_number = 1)
# Best_Sol, Best, history = DE.Solve()
#
# print(Best_Sol, Best)
# plt.plot(range(len(history)), history)
# plt.title("Fitness Function Vs. Number of Generations")
# plt.xlabel("Generation")
# plt.ylabel("Fitness")
# plt.xticks(np.arange(min(range(len(history))), max(range(len(history)))+1, 5))
# plt.show()


# Part 2:

bounds = np.array([[1, 100], [0.1, 1], [0.1, 0.1]])

history_of_histories = []
BestofBest = []
BestofBestSolution = []

Test = [[0.4, 0.1], [0.4, 0.35], [0.4, 0.5], [0.4, 0.7], [0.4, 1], [0.1, 0.7], [0.35, 0.7], [0.5, 0.7], [0.7, 0.7], [1, 0.7]]

for M_rate, CV_rate in Test:

    DE = Differential_Evolution(Population_size, NumberofParameters, NumberofGenerations, bounds, M_rate, CV_rate, 2)
    Best_Sol, Best, history = DE.Solve()
    history_of_histories.append(history)
    BestofBest.append(Best)
    BestofBestSolution.append(Best_Sol)

for i in range(10):
    plt.plot(range(len(history_of_histories[i])), list(-1 * np.array(history_of_histories[i])),
             label='M_R = ' + str(Test[i][0]) + ', CV_R = ' + str(Test[i][1]))

plt.title("Fitness Function Vs. Number of Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.xticks(np.arange(0, 31, 5))
plt.legend()
plt.show()

pps = []
zzs = []
for i in range(10):
    pps.append(str(Test[i][0]))
    zzs.append(str(Test[i][1]))

pps = np.array(pps)
pps = np.reshape(pps, (10, 1))
zzs = np.array(zzs)
zzs = np.reshape(zzs, (10, 1))
bestt = np.array(BestofBest)
bestt = np.round_(bestt, decimals=3)
bestt = np.reshape(bestt, (10, 1))
bestsol = np.array(BestofBestSolution)
bestsol = np.round_(bestsol, decimals=3)
header = np.array(['Mutation Rate', 'Crossover Rate', 'Optimal FItness Value', 'K', 'T1', 'T2'])
header = np.reshape(header, (1, 6))

tabel = np.concatenate((pps, zzs, -1*bestt, bestsol), axis = 1)
tabel = np.concatenate((header, tabel), axis = 0)
df = pd.DataFrame(data=tabel)

print(df)