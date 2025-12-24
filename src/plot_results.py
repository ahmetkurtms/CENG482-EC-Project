import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_project_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_history(dataset_name: str) -> pd.DataFrame:
    results_dir = os.path.join(get_project_root(), "results")
    history_file = os.path.join(results_dir, f"{dataset_name}_history.csv")
    return pd.read_csv(history_file)


def plot_fitness_convergence(df: pd.DataFrame, dataset_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['generation'], df['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
    ax.plot(df['generation'], df['avg_fitness'], 'r--', linewidth=1.5, alpha=0.7, label='Average Fitness')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness Value', fontsize=12)
    ax.set_title(f'GA Convergence - {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    if df['best_fitness'].max() / df['best_fitness'].min() > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Fitness Value (log scale)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def plot_violations_over_generations(df: pd.DataFrame, dataset_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['generation'], df['hard_violations'], 'g-', linewidth=2, marker='o',
            markersize=3, label='Hard Violations')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Number of Violations', fontsize=12)
    ax.set_title(f'Hard Constraint Violations - {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    zero_gen = df[df['hard_violations'] == 0]['generation'].min()
    if not pd.isna(zero_gen):
        ax.axvline(x=zero_gen, color='green', linestyle='--', alpha=0.7)
        ax.annotate(f'Zero violations\nat gen {int(zero_gen)}',
                   xy=(zero_gen, 0), xytext=(zero_gen + 10, df['hard_violations'].max() * 0.3),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   fontsize=10, color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    return fig


def generate_all_plots(dataset_name: str = "car-s-91"):
    print(f"\n{'='*50}")
    print("Generating Report Figures")
    print('='*50)

    df = load_history(dataset_name)
    print(f"✓ Loaded {len(df)} generations of data")

    figures_dir = os.path.join(get_project_root(), "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_fitness_convergence(
        df, dataset_name,
        os.path.join(figures_dir, f"{dataset_name}_fitness_convergence.png")
    )

    plot_violations_over_generations(
        df, dataset_name,
        os.path.join(figures_dir, f"{dataset_name}_violations.png")
    )

    print(f"\n{'='*50}")
    print(f"All figures saved to: {figures_dir}")
    print('='*50)


if __name__ == "__main__":
    generate_all_plots("car-s-91")
