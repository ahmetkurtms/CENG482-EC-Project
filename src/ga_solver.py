import random
from typing import Any, Dict, List, Tuple
from data_loader import load_data
from fitness import fitness_function

def initialize_population(exam_ids: List[int], population_size: int, rng: random.Random) -> List[List[int]]:
    population = []
    for _ in range(population_size):
        chromosome = exam_ids.copy()
        rng.shuffle(chromosome)
        population.append(chromosome)
    return population

def tournament_selection(evaluated: List[Dict[str, Any]], tournament_size: int, rng: random.Random) -> List[int]:
    contenders = rng.sample(evaluated, k=min(tournament_size, len(evaluated)))
    return min(contenders, key=lambda x: x['fitness'])['individual']

def order_crossover(parent1: List[int], parent2: List[int], rng: random.Random) -> List[int]:
    size = len(parent1)
    start, end = sorted(rng.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]
    fill_genes = [gene for gene in parent2 if gene not in child]
    fill_iter = iter(fill_genes)
    for i in range(size):
        if child[i] is None: child[i] = next(fill_iter)
    return child

def swap_mutation(chromosome: List[int], rng: random.Random) -> None:
    a, b = rng.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]

def insert_mutation(chromosome: List[int], rng: random.Random) -> None:
    i, j = rng.sample(range(len(chromosome)), 2)
    chromosome.insert(j, chromosome.pop(i))

def inversion_mutation(chromosome: List[int], rng: random.Random) -> None:
    i, j = sorted(rng.sample(range(len(chromosome)), 2))
    chromosome[i:j+1] = chromosome[i:j+1][::-1]

def scramble_mutation(chromosome: List[int], rng: random.Random) -> None:
    i, j = sorted(rng.sample(range(len(chromosome)), 2))
    segment = chromosome[i:j+1]
    rng.shuffle(segment)
    chromosome[i:j+1] = segment

MUTATION_FUNCS = {'swap': swap_mutation, 'insert': insert_mutation, 'inversion': inversion_mutation, 'scramble': scramble_mutation}

def evaluate_population(population: List[List[int]], conflict_matrix, exam_to_idx, students, max_slots: int, hard_penalty: float) -> List[Dict[str, Any]]:
    evaluated = []
    for individual in population:
        fitness, details = fitness_function(individual, conflict_matrix, exam_to_idx, students, max_slots, hard_penalty)
        evaluated.append({'individual': individual, 'fitness': fitness, 'details': details})
    return evaluated

def run_ga(data: Dict[str, Any], population_size: int = 80, generations: int = 150, crossover_rate: float = 0.9,
          mutation_rate: float = 0.2, tournament_size: int = 3, elitism: int = 2, max_slots: int = 40,
          hard_penalty: float = 1_000_000.0, mutation_type: str = 'swap', random_seed: int = None, verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(random_seed)
    mutate = MUTATION_FUNCS.get(mutation_type, swap_mutation)
    population = initialize_population(data['exam_ids'], population_size, rng)
    evaluated = evaluate_population(population, data['conflict_matrix'], data['exam_to_idx'], data['students'], max_slots, hard_penalty)
    best = min(evaluated, key=lambda x: x['fitness'])
    best = {'individual': best['individual'][:], 'fitness': best['fitness'], 'details': best['details']}
    history = []

    for gen in range(generations):
        evaluated.sort(key=lambda x: x['fitness'])
        best_gen = evaluated[0]
        avg_fitness = sum(e['fitness'] for e in evaluated) / len(evaluated)
        history.append({'generation': gen, 'best_fitness': best_gen['fitness'], 'avg_fitness': avg_fitness,
                        'slots': best_gen['details']['n_slots'], 'hard_violations': best_gen['details']['hard_violations']})
        if verbose and (gen == 0 or (gen + 1) % 10 == 0 or gen == generations - 1):
            print(f"Gen {gen + 1:3d} | best={best_gen['fitness']:.4f} avg={avg_fitness:.4f} slots={best_gen['details']['n_slots']} hard={best_gen['details']['hard_violations']}")
        
        next_population = [ind['individual'][:] for ind in evaluated[:elitism]]
        while len(next_population) < population_size:
            parent1, parent2 = tournament_selection(evaluated, tournament_size, rng), tournament_selection(evaluated, tournament_size, rng)
            if rng.random() < crossover_rate: child1, child2 = order_crossover(parent1, parent2, rng), order_crossover(parent2, parent1, rng)
            else: child1, child2 = parent1[:], parent2[:]
            if rng.random() < mutation_rate: mutate(child1, rng)
            if rng.random() < mutation_rate: mutate(child2, rng)
            next_population.extend([child1, child2])
        
        next_population = next_population[:population_size]
        evaluated = evaluate_population(next_population, data['conflict_matrix'], data['exam_to_idx'], data['students'], max_slots, hard_penalty)
        gen_best = min(evaluated, key=lambda x: x['fitness'])
        if gen_best['fitness'] < best['fitness']:
            best = {'individual': gen_best['individual'][:], 'fitness': gen_best['fitness'], 'details': gen_best['details']}
            
    return best, history

if __name__ == "__main__":
    data = load_data("car-s-91")
    best, history = run_ga(data, population_size=80, generations=150, crossover_rate=0.9, mutation_rate=0.2, tournament_size=3, elitism=2, max_slots=40, hard_penalty=1_000_000.0, random_seed=42, verbose=True)
    print(f"\nBest solution found\nFitness: {best['fitness']:.4f}\nSlots used: {best['details']['n_slots']}\nHard violations: {best['details']['hard_violations']}")
