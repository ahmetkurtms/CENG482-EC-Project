"""
experiment_mutations.py - Mutation Operator Comparison Experiment
------------------------------------------------------------------
This script runs the GA with different mutation operators and compares their performance.
Results are saved to CSV and plots are generated for the report.
"""

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

# Experiment parameters - same for all runs for fair comparison
EXPERIMENT_CONFIG = {
    'population_size': 100,
    'generations': 200,
    'crossover_rate': 0.9,
    'mutation_rate': 0.2,
    'tournament_size': 5,
    'elitism': 3,
    'max_slots': 40,
    'hard_penalty': 1_000_000.0,
    'random_seed': 42,  # Same seed for reproducibility
}

NUM_RUNS = 5  # Number of independent runs per mutation type


def run_experiment(data, mutation_type: str, run_id: int, seed: int):
    """Run GA with specified mutation type and return results."""
    config = EXPERIMENT_CONFIG.copy()
    config['mutation_type'] = mutation_type
    config['random_seed'] = seed

    start_time = time.time()
    best, history = run_ga(data, verbose=False, **config)
    elapsed = time.time() - start_time

    return {
        'mutation_type': mutation_type,
        'run_id': run_id,
        'seed': seed,
        'best_fitness': best['fitness'],
        'final_slots': best['details']['n_slots'],
        'hard_violations': best['details']['hard_violations'],
        'proximity_cost': best['details']['proximity_cost'],
        'time_seconds': elapsed,
        'history': history
    }


def run_all_experiments(data, num_runs: int = NUM_RUNS):
    """Run experiments for all mutation types."""
    results = []
    all_histories = {}

    base_seed = EXPERIMENT_CONFIG['random_seed']

    total_runs = len(MUTATION_TYPES) * num_runs
    current_run = 0

    for mutation_type in MUTATION_TYPES:
        print(f"\n{'='*60}")
        print(f"Testing mutation type: {mutation_type.upper()}")
        print('='*60)

        all_histories[mutation_type] = []

        for run_id in range(num_runs):
            current_run += 1
            seed = base_seed + run_id  # Different seed for each run

            print(f"  Run {run_id + 1}/{num_runs} (seed={seed})...", end=" ", flush=True)

            result = run_experiment(data, mutation_type, run_id, seed)
            results.append(result)
            all_histories[mutation_type].append(result['history'])

            print(f"fitness={result['best_fitness']:.4f}, "
                  f"violations={result['hard_violations']}, "
                  f"time={result['time_seconds']:.1f}s")

    return results, all_histories


def create_summary_table(results):
    """Create summary statistics table."""
    df = pd.DataFrame(results)

    summary = df.groupby('mutation_type').agg({
        'best_fitness': ['mean', 'std', 'min'],
        'proximity_cost': ['mean', 'std', 'min'],
        'hard_violations': ['mean', 'sum'],
        'time_seconds': ['mean', 'std']
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary


def plot_convergence_comparison(all_histories, save_path):
    """Plot convergence curves for all mutation types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'swap': 'blue', 'insert': 'red', 'inversion': 'green', 'scramble': 'orange'}

    # Plot 1: Best fitness convergence (average across runs)
    ax1 = axes[0]
    for mutation_type, histories in all_histories.items():
        # Average across runs
        all_best = np.array([[h['best_fitness'] for h in history] for history in histories])
        mean_best = np.mean(all_best, axis=0)
        std_best = np.std(all_best, axis=0)
        generations = range(1, len(mean_best) + 1)

        ax1.plot(generations, mean_best, color=colors[mutation_type],
                linewidth=2, label=mutation_type.capitalize())
        ax1.fill_between(generations, mean_best - std_best, mean_best + std_best,
                        color=colors[mutation_type], alpha=0.2)

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Convergence Comparison (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Final fitness box plot
    ax2 = axes[1]
    final_fitness_data = []
    labels = []
    for mutation_type in MUTATION_TYPES:
        histories = all_histories[mutation_type]
        final_values = [history[-1]['best_fitness'] for history in histories]
        final_fitness_data.append(final_values)
        labels.append(mutation_type.capitalize())

    bp = ax2.boxplot(final_fitness_data, labels=labels, patch_artist=True)
    for patch, mutation_type in zip(bp['boxes'], MUTATION_TYPES):
        patch.set_facecolor(colors[mutation_type])
        patch.set_alpha(0.6)

    ax2.set_xlabel('Mutation Operator', fontsize=12)
    ax2.set_ylabel('Final Best Fitness', fontsize=12)
    ax2.set_title('Final Fitness Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    return fig


def plot_violation_comparison(all_histories, save_path):
    """Plot hard violation elimination comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'swap': 'blue', 'insert': 'red', 'inversion': 'green', 'scramble': 'orange'}
    linestyles = {'swap': '-', 'insert': '--', 'inversion': '-.', 'scramble': ':'}

    for mutation_type, histories in all_histories.items():
        # Average across runs
        all_violations = np.array([[h['hard_violations'] for h in history] for history in histories])
        mean_violations = np.mean(all_violations, axis=0)
        generations = range(1, len(mean_violations) + 1)

        ax.plot(generations, mean_violations, color=colors[mutation_type],
               linestyle=linestyles[mutation_type], linewidth=2,
               label=mutation_type.capitalize())

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Hard Violations (avg)', fontsize=12)
    ax.set_title('Hard Constraint Violation Elimination', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    return fig


def save_results(results, summary, output_dir):
    """Save all experiment results as CSV files."""
    # Save detailed results
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'history'}
                               for r in results])
    results_file = os.path.join(output_dir, "mutation_comparison_results.csv")
    df_results.to_csv(results_file, index=False)
    print(f"✓ Saved: {results_file}")

    # Save summary
    summary_file = os.path.join(output_dir, "mutation_comparison_summary.csv")
    summary.to_csv(summary_file)
    print(f"✓ Saved: {summary_file}")


def main():
    print("="*60)
    print("MUTATION OPERATOR COMPARISON EXPERIMENT")
    print("="*60)

    # Load data
    dataset_name = "car-s-91"
    print(f"\nLoading dataset: {dataset_name}")
    data = load_data(dataset_name)
    print(f"✓ {data['n_exams']} exams, {data['n_students']} students")

    # Print experiment configuration
    print(f"\nExperiment Configuration:")
    print(f"  - Mutation types: {MUTATION_TYPES}")
    print(f"  - Runs per type: {NUM_RUNS}")
    print(f"  - Generations: {EXPERIMENT_CONFIG['generations']}")
    print(f"  - Population size: {EXPERIMENT_CONFIG['population_size']}")

    # Run experiments
    print(f"\nStarting experiments ({len(MUTATION_TYPES) * NUM_RUNS} total runs)...")
    start_time = time.time()

    results, all_histories = run_all_experiments(data, NUM_RUNS)

    total_time = time.time() - start_time
    print(f"\n✓ All experiments completed in {total_time:.1f} seconds")

    # Create summary
    summary = create_summary_table(results)

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(summary.to_string())

    # Create output directory
    output_dir = os.path.join(get_project_root(), "results", "mutation_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    plot_convergence_comparison(
        all_histories,
        os.path.join(output_dir, "mutation_convergence_comparison.png")
    )

    plot_violation_comparison(
        all_histories,
        os.path.join(output_dir, "mutation_violation_comparison.png")
    )

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    save_results(results, summary, output_dir)

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)

    return results, summary


if __name__ == "__main__":
    main()
