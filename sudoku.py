import datetime
from numpy import random
import random as rd
import time
import sys
import numpy as np
from matplotlib import pyplot as plt

grid = [
[4, 0, 0, 0, 0, 5, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 9, 8],
[3, 0, 0, 0, 8, 2, 4, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 8, 0],
[9, 0, 3, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 3, 0, 6, 7, 0],
[0, 5, 0, 0, 0, 9, 0, 0, 0],
[0, 0, 0, 2, 0, 0, 9, 0, 7],
[6, 4, 0, 3, 0, 0, 0, 0, 0],
]

grid = [[0, 4, 3, 0, 8, 0, 2, 5, 0],
       [6, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 9, 4],
       [9, 0, 0, 0, 0, 4, 0, 7, 0],
       [0, 0, 0, 6, 0, 8, 0, 0, 0],
       [0, 1, 0, 2, 0, 0, 0, 0, 3],
       [8, 2, 0, 5, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 5],
       [0, 3, 4, 0, 9, 0, 7, 1, 0]]

grid = np.array(grid)



def generateRandomPopulation(grid, n):
    pop = []
    for i in range(n):
        pop.append(generateRandomIndividual(grid,i))
    return pop


def generateRandomIndividual(board,z):
    temp = random.randint(1,9,size=(9,9))
    for i in range(9):
        for j in range(9):
            if fixedValues[i][j] == 1:
                temp[i][j] = board[i][j]
    return temp

def generateFixedValues(board):
    fixedValues = np.zeros((9,9))
    for x in range(9):
        for y in range(9):
            if int(board[x][y]) > 0:
                fixedValues[x][y] = 1
    return fixedValues

def mutate(board, mutpb):
    # num cells to mutate
    num_mut = 3

    row_count = 0
    for r in range(9):
        if random.random() < mutpb:

            flag = False
            back_out_count = 20
            while flag==False and back_out_count > 0:
                temp = random.randint(0,8)
                if fixedValues[r][temp] != 1:
                    board[r][temp] = np.random.randint(1,9)
                    #print("Mutation Occured")
                    flag = True
                else:
                    back_out_count -= 1
        row_count += 1

    return board

def mutate_alternate(board , mutpb):
    for i in range(9):
        for j in range(9):
            if fixedValues[i][j] != 1:
                if random.random() < mutpb:
                    board[i][j] = random.randint(1,9)

    return board

# 2 point cross over strategy
def crossover(board, rowone, rowtwo):
    pointone = 3
    pointtwo = 6



    for i in range(pointone):
        if fixedValues[rowone][i] != 1:
            board[rowone][i] = board[rowtwo][i]
        if fixedValues[rowtwo][i] != 1:
            board[rowtwo][i] = board[rowone][i]
    for i in range(pointtwo,9):
        if fixedValues[rowone][i] != 1:
            board[rowone][i] = board[rowtwo][i]
        if fixedValues[rowtwo][i] != 1:
            board[rowtwo][i] = board[rowone][i]
    return board


def fitnessFunction_toZero(board):
    score = 0
    temp = []
    for row in board:
        temp = []
        for i in row:
            if i not in temp:
                temp.append(i)
            else:
                score += 1
    for col in board.T:
        temp = []
        for i in col:
            if i not in temp:
                temp.append(i)
            else:
                score += 1
    temp = []
    for i in range(0, 3):
        for j in range(0, 3):
            sub = np.array(board[3 * i:3 * i + 3, 3 * j:3 * j + 3])
            sub = sub.flatten()
            for i in sub:
                if i not in temp:
                    temp.append(i)
                else:
                    score += 1

    return score

def fitnessFunction(board):
    score = 0
    for row in board:
        score += len(np.unique(row))
    for col in board.T:
        score += len(np.unique(col))
    for i in range(0, 3):
        for j in range(0, 3):
            sub = board[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            score += len(np.unique(sub))

    return score

def crossover_v2(p1, p2 , cxpb):
    c1 , c2 = p1.copy() , p2.copy()


    rownum1 = random.randint(1,9)
    rownum2 = random.randint(100,300)

    rownum2 = (rownum2%9) -1

    if random.random() < cxpb:
        for j in range(9):
            if fixedValues[rownum1][j] != 1:
                c1[rownum1][j] = p2[rownum2][j]
            if fixedValues[rownum2][j] != 1:
                c2[rownum2][j] = p1[rownum1][j]

    return [c1,c2]

def randomSample(fitness,sample_size):
    temp = np.random.choice(fitness, size=sample_size, replace=False)
    return np.argmax(temp)

def randomSample_toZero(fitness, sample_size):
    temp = np.random.choice(fitness, sample_size)
    return np.argmin(temp)

def tournament(fitness):
    sample = random.randint(0,len(fitness))
    for i in random.randint(0,len(fitness)):
        if fitness[i] > fitness[sample]:
            sample = i
    return sample

def tournament_tozero(fitness):
    sample = random.randint(0, len(fitness))
    for i in random.randint(0, len(fitness),1):
        if fitness[i] < fitness[sample]:
            sample = i
    return sample

def mid_algorithm_randomize_population(population, n):
    pop_copy = population
    for i in range(n):
        random_individual = random.randint(0,len(pop_copy))
        pop_copy[random_individual] = generateRandomIndividual(grid,0)
    return pop_copy

def late_algorithm_randomize_population(population, n):

    for i in range(n):
        random_individual = random.randint(0, len(population))
        random_individual_board = population[random_individual]

        random_row_start = random.randint(0,8)
        random_col_start = random.randint(0,8)

        random_row_remainder = random_row_start - random_row_start%3
        random_col_remainder = random_col_start - random_col_start%3

        for p in range(3):
            for q in range(3):
                random_individual_board[p][q] = random.randint(1,9)

        population[random_individual] = random_individual_board
        return population


def open_textfile_return_grid(textfile):
    f = open(textfile,"r")
    board = []
    for lines in f:
        temp = []
        for i in lines:
            if i != " ":
                temp.append(i)
        board.append(temp)

    return np.array(board)




# Start generations
 # grid filled with zeros and ones where fixed number positions are

#for gen in range(N_GEN):
random.seed(666666)
gen = 0

def GA_to243():
    gen = 0
    N_GEN = 10000
    pop_size = 1000

    MUTPB = 0.005  # mutation probability | would recommend starting at 0.1 and fidgeting from there
    CXPB = 0.1  # Crossover probability | would recommend starting at 0.7 and fidgeting from there
    num_parents = 20
    stagnation = 0  # stagnation counter
    stagnent_limit = 50  # amount of times fitness can be the same before increasing mutation chance
    NUM_NEW_POP = 1
    NUM_NEW_BLOCK = 1
    crossover_sample_size = 400  # use for tournament selection

    currmax = 5
    prevmax = -1
    population = generateRandomPopulation(grid,pop_size)
    while currmax != 243:
        # fitness test whole population
        print("-----GEN %i-----" % gen)
        fitness = np.array([fitnessFunction(b) for b in population])

        parents = [] * num_parents
        # print("---CHECKPOINT %i ----" % 1)

        # elitism strategy
        fitness_temp = fitness
        for i in range(num_parents):
            parents.append(fitness_temp.argmax())
            fitness_temp = np.delete(fitness_temp, fitness_temp.argmax())

        children = []
        for l in range(int(pop_size / 2)):  # might have to change to pop_size/2

            indexone = parents[(l + 1) % num_parents]
            indextwo = parents[l % num_parents]
            # p1 = population[randomSample(fitness, crossover_sample_size)]   # use for tournament selection V1
            # p2 = population[randomSample(fitness, crossover_sample_size)]   # use for tournament selection V1

            p1 = population[tournament(fitness)]
            p2 = population[tournament(fitness)]
            #p1, p2 = population[indexone], population[indextwo]   # use for elitism selection
            for c in crossover_v2(p1, p2, CXPB):
                c = mutate_alternate(c, MUTPB)

                children.append(c)

        #stagnation solution: generate random board and introduce to pop
        if(population[fitness.argmax()]) > 20:
            population = mid_algorithm_randomize_population(population,NUM_NEW_POP)
        else:
            population = late_algorithm_randomize_population(population,NUM_NEW_BLOCK)

        # stagnation



        print(stagnation)
        currmax = max(fitness)
        if (currmax == prevmax):
            stagnation += 1
        else:
            stagnation = 0
            prevmax = currmax

        if (stagnation >= stagnent_limit):
            MUTPB = 1.0
            print("Super mutation")
            stagnation = stagnent_limit / 4
        else:
            MUTPB = 0.2

        population = children
        print(np.max(fitness))
        gen += 1
    return population, fitness

#200 pop, 2 parents, max +prev max , 0.05, 0.5 decent result
# try to get to 0, new heuristic
def GA_tozero(gen_limit,grid):

    ############################################################################
                                #SETTINGS#
    ############################################################################
    gen = 0
    N_GEN = 10000
    pop_size = 40
    fixedValues = generateFixedValues(grid)  # grid filled with zeros and ones where fixed number positions are
    MUTPB = 0.04  # mutation probability | would recommend starting at 0.1 and fidgeting from there
    CXPB = 0.5  # Crossover probability | would recommend starting at 0.7 and fidgeting from there
    num_parents = 2
    stagnation = 0  # stagnation counter
    stagnent_limit = 50  # amount of times fitness can be the same before increasing mutation chance
    NUM_NEW_POP = 1
    NUM_NEW_BLOCK = 1
    crossover_sample_size = 400  # use for tournament selection
    avg = []
    avg2 = []
    avg_count = 0
    currmax = 5
    prevmax = -1


    #############################################################################
    population = generateRandomPopulation(grid, pop_size)
    while gen < int(gen_limit):
        print("-----GEN %i-----" % gen)
        fitness = np.array([fitnessFunction_toZero(b) for b in population])

        parents = [] * num_parents
        # print("---CHECKPOINT %i ----" % 1)

        avg.append(int(sum(fitness) / len(fitness)))
        avg2.append(int(sum(fitness) / len(fitness)))
        avg_count += 1
        avg_sum = 0
        if avg_count > 2050:
            for i in range(2000):
                avg_sum += avg[i]
            if avg[avg_count-1] in range(int(avg_sum/2000),int(avg_sum/2000)+1):
                avg = []
                avg_count = 0
                population = generateRandomPopulation(grid,pop_size)

        # elitism strategy
        fitness_temp = fitness
        for i in range(num_parents):
            parents.append(fitness_temp.argmin())
            fitness_temp = np.delete(fitness_temp, fitness_temp.argmin())

        p_max = random.randint(1, pop_size)
        prevmax = 0;

        children = []
        for l in range(pop_size):  # might have to change to pop_size/2

            indexone = parents[(l+1) % num_parents]
            indextwo = parents[l % num_parents]
            #p1 = population[randomSample_toZero(fitness, crossover_sample_size)]   # use for tournament selection V1
            #p2 = population[randomSample_toZero(fitness, crossover_sample_size)]   # use for tournament selection V1

            #p1 = population[tournament_tozero(fitness)]   #tournament v2
            #p2 = population[tournament_tozero(fitness)]    #tournament v2
            p1, p2 = population[indexone], population[indextwo]   # use for elitism selection

            prev_p_max = p_max
            p_max = fitness.argmin()


            #p1, p2 = population[p_max], population[prev_p_max]
            for c in crossover_v2(p1, p2, CXPB):
                #c = mutate(c, MUTPB)
                c = mutate_alternate(c,MUTPB)
                children.append(c)

        if fitness.argmin() > 31:
            #if random.random() > MUTPB:
            population = mid_algorithm_randomize_population(population, NUM_NEW_POP)
        else:
            #if random.random() > MUTPB:
            population = late_algorithm_randomize_population(population, NUM_NEW_BLOCK)

        # stagnation
#        print(stagnation)
#        currmax = max(fitness)
#        if (currmax == prevmax):
#            stagnation += 1
#        else:
#            stagnation = 0
#            prevmax = currmax

 #       if (stagnation >= stagnent_limit):
#            MUTPB = 1.0
#            print("Super mutation")
#            stagnation = stagnent_limit / 4
#        else:
#            MUTPB = 0.01

        population = children
        #print(population[fitness.argmin()])
        print("\n================================\n")
        print(np.min(fitness))
        gen += 1
    return population , fitness, avg2



#temp_pop, temp_fitness = GA_to243()
#temp_pop, temp_fitness = GA_tozero()
#print(temp_pop[temp_fitness.argmax()])

#to initialize the fixed values array

grid = open_textfile_return_grid(sys.argv[1])
fixedValues = generateFixedValues(grid)

population, fitness,avg = GA_tozero(sys.argv[2], grid)
print("==================FINAL==============")
print(population[fitness.argmin()])
print("\n\n-----------------------------\n\n")
print(fitness.argmin())

f = open("output1.txt","w")

for i in population[fitness.argmin()]:
    temp =""
    for j in i:
        temp = temp + str(j) + " "
    f.write(temp+'\n')

f.close()
f = open("output2.txt", "w")

for i in avg:
    f.write(str(i) + '\n')

f.close()

plt.plot(avg)
