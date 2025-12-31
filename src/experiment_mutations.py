import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_data, get_project_root
from ga_solver import run_ga

MUTATION_TYPES = ['swap', 'insert', 'inversion', 'scramble']
EXPERIMENT_CONFIG = {
    'population_size': 100, 'generations': 200, 'crossover_rate': 0.9, 'mutation_rate': 0.2,
    'tournament_size': 5, 'elitism': 3, 'max_slots': 40, 'hard_penalty': 1_000_000.0, 'random_seed': 42,
}
NUM_RUNS = 5

def run_experiment(data, mutation_type: str, run_id: int, seed: int):
    config = EXPERIMENT_CONFIG.copy()
    config['mutation_type'] = mutation_type
    config['random_seed'] = seed
    start_time = time.time()
    best, history = run_ga(data, verbose=False, **config)
    return {
        'mutation_type': mutation_type, 'run_id': run_id, 'seed': seed,
        'best_fitness': best['fitness'], 'final_slots': best['details']['n_slots'],
        'hard_violations': best['details']['hard_violations'], 'proximity_cost': best['details']['proximity_cost'],
        'time_seconds': time.time() - start_time, 'history': history
    }

def run_all_experiments(data, num_runs=NUM_RUNS):
    results, all_histories = [], {}
    base_seed = EXPERIMENT_CONFIG['random_seed']
    for mutation_type in MUTATION_TYPES:
        print(f"\n{'='*60}\nTesting mutation type: {mutation_type.upper()}\n{'='*60}")
        all_histories[mutation_type] = []
        for run_id in range(num_runs):
            seed = base_seed + run_id
            print(f"  Run {run_id + 1}/{num_runs} (seed={seed})...", end=" ", flush=True)
            result = run_experiment(data, mutation_type, run_id, seed)
            results.append(result)
            all_histories[mutation_type].append(result['history'])
            print(f"fitness={result['best_fitness']:.4f}, violations={result['hard_violations']}, time={result['time_seconds']:.1f}s")
    return results, all_histories

def create_summary_table(results):
    df = pd.DataFrame(results)
    summary = df.groupby('mutation_type').agg({
        'best_fitness': ['mean', 'std', 'min'], 'proximity_cost': ['mean', 'std', 'min'],
        'hard_violations': ['mean', 'sum'], 'time_seconds': ['mean', 'std']
    }).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary

def plot_convergence_comparison(all_histories, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'swap': 'blue', 'insert': 'red', 'inversion': 'green', 'scramble': 'orange'}
    
    ax1 = axes[0]
    for m_type, histories in all_histories.items():
        all_best = np.array([[h['best_fitness'] for h in history] for history in histories])
        mean_best, std_best = np.mean(all_best, axis=0), np.std(all_best, axis=0)
        gens = range(1, len(mean_best) + 1)
        ax1.plot(gens, mean_best, color=colors[m_type], linewidth=2, label=m_type.capitalize())
        ax1.fill_between(gens, mean_best - std_best, mean_best + std_best, color=colors[m_type], alpha=0.2)
    ax1.set_xlabel('Generation'); ax1.set_ylabel('Best Fitness'); ax1.set_title('Convergence Comparison'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')

    ax2 = axes[1]
    final_fitness_data = [[h[-1]['best_fitness'] for h in all_histories[m]] for m in MUTATION_TYPES]
    bp = ax2.boxplot(final_fitness_data, labels=[m.capitalize() for m in MUTATION_TYPES], patch_artist=True)
    for patch, m in zip(bp['boxes'], MUTATION_TYPES):
        patch.set_facecolor(colors[m]); patch.set_alpha(0.6)
    ax2.set_xlabel('Mutation Operator'); ax2.set_ylabel('Final Best Fitness'); ax2.set_title('Final Fitness Distribution'); ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_violation_comparison(all_histories, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'swap': 'blue', 'insert': 'red', 'inversion': 'green', 'scramble': 'orange'}
    linestyles = {'swap': '-', 'insert': '--', 'inversion': '-.', 'scramble': ':'}
    
    for m_type, histories in all_histories.items():
        mean_violations = np.mean([[h['hard_violations'] for h in history] for history in histories], axis=0)
        ax.plot(range(1, len(mean_violations)+1), mean_violations, color=colors[m_type], linestyle=linestyles[m_type], linewidth=2, label=m_type.capitalize())
        
    ax.set_xlabel('Generation'); ax.set_ylabel('Hard Violations (avg)'); ax.set_title('Hard Constraint Violation Elimination'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def save_results(results, summary, output_dir):
    pd.DataFrame([{k: v for k, v in r.items() if k != 'history'} for r in results]).to_csv(os.path.join(output_dir, "mutation_comparison_results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "mutation_comparison_summary.csv"))

def main():
    print("="*60 + "\nMUTATION OPERATOR COMPARISON EXPERIMENT\n" + "="*60)
    data = load_data("car-s-91")
    print(f"Loading dataset: car-s-91\n✓ {data['n_exams']} exams, {data['n_students']} students\n\nStarting experiments...")
    
    results, all_histories = run_all_experiments(data, NUM_RUNS)
    summary = create_summary_table(results)
    
    print("\n" + "="*60 + "\nSUMMARY STATISTICS\n" + "="*60 + "\n" + summary.to_string())
    output_dir = os.path.join(get_project_root(), "results", "mutation_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_convergence_comparison(all_histories, os.path.join(output_dir, "mutation_convergence_comparison.png"))
    plot_violation_comparison(all_histories, os.path.join(output_dir, "mutation_violation_comparison.png"))
    save_results(results, summary, output_dir)
    print("\n" + "="*60 + f"\nEXPERIMENT COMPLETE\nResults saved to: {output_dir}\n" + "="*60)

if __name__ == "__main__":
    main()
