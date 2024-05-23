import numpy as np
import matplotlib.pyplot as plt

bounds = [-3.0, 3.0]
n_iterations = 1000
n_population = 100
n_genes = 16
max_stagnation = 100
rate_crossover = 0.9
rate_mutation =  0.1

def adaptability(x):
	return 2.0 ** x * np.sin(10 * x)

def fitness(decoded): 
    return [adaptability(x) for x in decoded]

def decode(population):
    min_val, max_val = bounds
    decimals = []

    for i in range(len(population)):
        string = ''.join([str(gene) for gene in population[i]])
        decimal = int(string, 2)
        decoded = min_val + decimal * (max_val - min_val) / (2 ** n_genes - 1) 
        decimals.append(decoded)

    return decimals

def minmax(population, f_type):
    best = 0

    if f_type == 'min':
        best = np.min(population)
    else:
        best = np.max(population)
        
    index = population.index(best)
    return [best, index]

def selection(population, fitness, f_type ):
    selected = []

    for _ in range(len(population)):
        indices = np.random.choice(len(population), size = 3, replace = False)
        t = [population[i] for i in indices]
        f = [fitness[i] for i in indices]

        best_value, best_index = minmax(f, f_type)
        selected.append(t[best_index])
    return selected

def crossover(selected):
    children = []

    for i in range(0, len(selected), 2):
        parent1, parent2 = selected[i], selected[i + 1]
        
        if np.random.rand() < rate_crossover:
            crossover_point = np.random.randint(1, n_genes - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        else:
            child1, child2 = parent1, parent2
        
        children.append(child1)
        children.append(child2)

    return children

def mutation(children):
    for child in children:
        for gene in range(n_genes):
            if np.random.rand() < rate_mutation:
                child[gene] = 1 - child[gene] 
    return children

def genetic_algorithm(f_type):
    population = np.random.randint(0, 2, (n_population, n_genes))
    x = decode(population)
    f = fitness(x)

    best, index = minmax(f, f_type)
    best_fitness = best
    best_chromosome = x[index]
    stagnation = 0
    print('Initial best x = ', best_chromosome)
    print('Initial best y = ', best_fitness)

    for i in range(n_iterations):
        if stagnation >= max_stagnation:
            break

        selected = selection(population, f, f_type)
        children = crossover(selected)
        population = mutation(children)

        x = decode(population)
        f = fitness(x)
        current_best, current_index = minmax(f, f_type)

        if abs(current_best) > abs(best_fitness):
            best_fitness = current_best
            best_chromosome = x[current_index]
            stagnation = 0
        else:
            stagnation += 1

        print(f'Iteration {i + 1}: best x = {best_chromosome}, best y = {best_fitness}, stagnation = {stagnation}')

    print('Final best x =', best_chromosome)
    print('Final best y =', best_fitness)

genetic_algorithm('max')
# genetic_algorithm('min')

x = np.linspace(-2, 5, 1000)

plt.plot(x, adaptability(x))
plt.xlabel('x')
plt.ylabel('y')
plt.title('2.0 ** x * math.sin(10 * x)')
plt.show()
