from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def selekcja_ruletkowa(population, population_fitness, n_selection):
    total_fitness = sum(population_fitness)

    selection_probabilities = [fitness / total_fitness for fitness in population_fitness]

    selected_indices = random.choices(range(len(population_fitness)), weights=selection_probabilities, k=n_selection)

    selected_individuals = [population[index] for index in selected_indices]

    return selected_individuals, selected_indices

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 2

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

    population_fitness =[]

    for individual in population:
        population_fitness.append(fitness(items, knapsack_max_capacity, individual))

    top_population, indexes = selekcja_ruletkowa(population, population_fitness, n_selection)

    population_sorted = sorted(population, key=lambda p: fitness(items, knapsack_max_capacity, p))

    population_without_top = []
    for i in range(0, len(population) - 1):
        if i not in indexes:
            population_without_top.append(population[i])

    elite_population = population_sorted[-n_elite:]

    new_population = []
    for i in range(0, ((population_size-n_elite)//2)):

        first_row_half1 = random.choice(top_population)[:len(population[0]) // 2]
        first_row_half2 = random.choice(top_population)[len(population[0]) // 2:]
        second_row_half1 = random.choice(top_population)[:len(population[0]) // 2]
        second_row_half2 = random.choice(top_population)[len(population[0]) // 2:]


        child1 = (first_row_half1 + second_row_half2)
        child2 = (second_row_half1 + first_row_half2)
        mutation = random.randint(0, len(child1) - 1)
        child1[mutation] = not child1[mutation]
        elite_population.append(child1)
        mutation = random.randint(0, len(child2) - 1)
        child2[mutation] = not child2[mutation]
        elite_population.append(child2)


    population = elite_population

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
