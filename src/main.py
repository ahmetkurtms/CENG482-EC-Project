import os
import time
from typing import Dict
from data_loader import load_data, get_project_root
from decoder import get_slot_exams
from ga_solver import run_ga

def print_summary(dataset_name: str, data: Dict, result: Dict, elapsed: float, generations: int):
    print("\n" + "="*60 + "\nOPTIMIZATION RESULTS\n" + "="*60)
    print(f"Dataset: {dataset_name}\nTotal exams: {data['n_exams']}\nTotal students: {data['n_students']}")
    print("-"*60 + f"\nGenerations: {generations}\nTime elapsed: {elapsed:.2f} seconds\n" + "-"*60)
    print(f"Best Fitness: {result['fitness']:.4f}")
    print(f"Slots used: {result['details']['n_slots']}")
    print(f"Hard violations: {result['details']['hard_violations']}")
    print(f"Proximity cost: {result['details']['proximity_cost']:.4f}\n" + "="*60)

def save_results(dataset_name: str, result: Dict, history: list, output_dir: str = None):
    if output_dir is None: output_dir = os.path.join(get_project_root(), "results")
    os.makedirs(output_dir, exist_ok=True)
    slot_file = os.path.join(output_dir, f"{dataset_name}_solution.txt")
    with open(slot_file, 'w') as f:
        f.write(f"# Best solution for {dataset_name}\n# Fitness: {result['fitness']:.4f}\n")
        f.write(f"# Slots: {result['details']['n_slots']}\n# Violations: {result['details']['hard_violations']}\n")
        f.write(f"# Proximity: {result['details']['proximity_cost']:.4f}\n\n")
        slot_exams = get_slot_exams(result['details']['slot_assignment'])
        for slot_num in sorted(slot_exams.keys()):
            f.write(f"Slot {slot_num + 1}: {' '.join(map(str, sorted(slot_exams[slot_num])))}\n")
    print(f"Solution saved to: {slot_file}")
    
    history_file = os.path.join(output_dir, f"{dataset_name}_history.csv")
    with open(history_file, 'w') as f:
        f.write("generation,best_fitness,avg_fitness,slots,hard_violations\n")
        for h in history:
            f.write(f"{h['generation']+1},{h['best_fitness']:.6f},{h['avg_fitness']:.6f},{h['slots']},{h['hard_violations']}\n")
    print(f"History saved to: {history_file}")

def main():
    print("="*60 + "\nCENG482 - Exam Scheduling Genetic Algorithm\n" + "="*60)
    dataset_name = "car-s-91"
    print(f"\nLoading dataset: {dataset_name}")
    data = load_data(dataset_name)
    print(f"✓ {data['n_exams']} exams, {data['n_students']} students\n\nRunning Genetic Algorithm...")
    
    start_time = time.time()
    best, history = run_ga(
        data, population_size=100, generations=200, crossover_rate=0.9, mutation_rate=0.2,
        tournament_size=5, elitism=3, max_slots=40, hard_penalty=1_000_000.0,
        mutation_type='swap', random_seed=42, verbose=True
    )
    elapsed = time.time() - start_time
    print_summary(dataset_name, data, best, elapsed, len(history))
    save_results(dataset_name, best, history)
    return best, history

if __name__ == "__main__":
    main()
