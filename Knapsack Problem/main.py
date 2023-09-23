import random
import pandas as pd
import itertools
import numpy as np


def string_to_list(string):

  string_list = string.strip('[]').split()

  float_list = [float(element) for element in string_list]

  return float_list

class Individual():
  def __init__(self):
    self.items: list[int] = [random.randint(0, 1) for i in range(5)]
    self.fitness: float = 0.0

  def __eq__(self, rs) -> bool:
    return self.fitness == rs.fitness

  def __lt__(self, rs) -> bool:
    return self.fitness < rs.fitness

  def __str__(self) -> str:
    return "Fitness: " + str(self.fitness) + ", Items: " + str(self.items)

  def __repr__(self) -> str:
    return str(self)

# Calculate the fitness based on the acheived weight of the knapsack vs maximum.
# If we've passed maximum capacity weight it as negative by the same amount. (TODO Adjust this)
# 0 is optimal in this fitness function.
def calculate_fitness(ind, prices, weights, capacity) -> float:
  price = 0.0
  weight = 0.0


  # for i in ind.items:
  for i, choice in enumerate(ind.items):

    if choice == 1:
      price += prices[i]
      weight += weights[i]

  if weight > capacity:
    ind.fitness = 0.0
    return 0.0

  ind.fitness = price
  return price


def crossover(parent1, parent2, cross_rate):
  # Take a portion of the bits to cross over between parents
  c_len = int(5 * cross_rate) # number of bits to cross
  ind1 = Individual()
  ind2 = Individual()
  ind1.items = parent1.items[:c_len] + parent2.items[c_len:]
  ind2.items = parent2.items[:c_len] + parent1.items[c_len:]

  return ind1, ind2


def mutation(child, mut_rate) -> Individual:
  for i, bit in enumerate(child.items):
    if random.random() < mut_rate:
      child.items[i] = (bit + 1) % 2

  return child


def tournament(population: list) -> list:
  left = population[:len(population)//2]
  right = population[len(population)//2:]
  out = []
  for (l, r) in zip(left, right):
    if l.fitness > r.fitness:
      out.append(l)
    else:
      out.append(r)

  return out


def genetic_algorithm(data, population_size, num_generations, mut_rate, cross_rate, tournament_size):
  # Generate random initial population of the given size with fitness.
  # (individual, fitness)
  population: list[Individual] = [Individual() for i in range(population_size)]

  for iteration in range(num_generations):
    # Choose Random Tournament Participants
    participant_indices = random.sample(range(0, len(population) - 1), tournament_size)
    participants = [population[i] for i in participant_indices]
    [population.pop(i) for i in sorted(participant_indices, reverse=True)]

    # Evaluate fitness of the Participants
    for ind in participants:
      calculate_fitness(ind, data['Prices'], data['Weights'], data['Capacity'])

    # Run the tournament
    while len(participants) > 2:
      participants = tournament(participants)

    # Reproduce the winners;
    child1, child2 = crossover(participants[0], participants[1], cross_rate)
    while len(population) < population_size:
      population.extend([mutation(child1, mut_rate), mutation(child2, mut_rate)])

  for ind in population:
    calculate_fitness(ind, data['Prices'], data['Weights'], data['Capacity'])
  best = sorted(population, reverse=True)[0]

  # print("Solved: ", [i for i in sorted(population, reverse=True)])

  return best.fitness, best.items




url = "https://raw.githubusercontent.com/AlexanderDefuria/CSI4106/main/datasets/knapsack_5_items.csv"
dataset = pd.read_csv(url)
dataset = dataset.dropna()
dataset.Weights = dataset.Weights.apply(lambda x : string_to_list(x))
dataset.Prices = dataset.Prices.apply(lambda x : string_to_list(x))
dataset['Best picks'] = dataset['Best picks'].apply(lambda x : string_to_list(x))
populations_sizes = [25, 50, 100, 200]
num_generations = [10, 20, 40, 80]
mut_rates = [0.005, 0.01, 0.04, 0.1, 0.3]
cross_rates = [0, 0.2, 0.4]
tournament_sizes = [4, 11]
sample_size = [500]

with open('output.csv', 'w') as f:
  f.write('populations_sizes,num_generations,mut_rates,cross_rates,tournament_sizes,sample_size,accuracy\n')
  for p in itertools.product(*[populations_sizes, num_generations, mut_rates, cross_rates, tournament_sizes, sample_size]):
    solutions_ga = []
    average_diff = []
    actual_solutions = []
    for i, row in dataset.iterrows():
        target = row['Best price']
        # solution, indexes = genetic_algorithm(row, population_size = 100, num_generations = 50, mut_rate = 0.02, cross_rate = 0.4, tournament_size = 5)
        solution, indexes = genetic_algorithm(row, population_size = p[0], num_generations = p[1], mut_rate = p[2], cross_rate = p[3], tournament_size = p[4])
        solutions_ga.append(1 if target == solution else 0)

        if i == p[5]:
          break
    out = ','.join([str(x) for x in [p[0], p[1], p[2], p[3], p[4], p[5], np.mean(solutions_ga)]])
    print(out)
    f.write(out + '\n')
