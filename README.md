# CENG482-EC-Project

Genetic Algorithm-based solution to the **Exam Timetabling Problem** — CENG482 (Evolutionary Computation) Term Project

## Problem Description

The Exam Timetabling Problem involves scheduling exams into time slots while:
- **Hard Constraint**: No student should have two exams at the same time
- **Soft Constraint**: Minimize proximity conflicts (exams close together for the same student)

The fitness function uses Carter's proximity cost with weights `[16, 8, 4, 2, 1]` for gaps of 1-5 slots.

## Project Structure

```
├── data/                  # Dataset files
│   ├── car-s-91.crs       # Course information
│   └── car-s-91.stu       # Student enrollments
├── results/               # Output files
│   ├── *_solution.txt     # Best solution found
│   └── *_history.csv      # Fitness history per generation
├── src/
│   ├── main.py            # Main entry point
│   ├── ga_solver.py       # Genetic Algorithm implementation
│   ├── fitness.py         # Fitness function
│   ├── decoder.py         # Chromosome decoder (greedy slot assignment)
│   └── data_loader.py     # Data loading utilities
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/ahmetkurtms/CENG482-EC-Project.git
cd CENG482-EC-Project

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
cd src
python main.py
```

This will:
1. Load the `car-s-91` dataset
2. Run the Genetic Algorithm with default parameters
3. Save results to `results/` directory

## Parameters

You can adjust the following GA parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Number of individuals in population |
| `generations` | 200 | Number of generations to run |
| `crossover_rate` | 0.9 | Probability of crossover |
| `mutation_rate` | 0.2 | Probability of mutation |
| `tournament_size` | 5 | Tournament selection size |
| `elitism` | 3 | Number of best individuals preserved |
| `max_slots` | 40 | Maximum time slots available |
| `hard_penalty` | 1,000,000 | Penalty for hard constraint violations |
| `mutation_type` | 'swap' | Mutation operator type |
| `random_seed` | 42 | Random seed for reproducibility |

## Output

After execution, results are saved in the `results/` folder:

- **`car-s-91_solution.txt`**: Final exam schedule with slot assignments
- **`car-s-91_history.csv`**: Generation-by-generation fitness tracking

## Dataset

The project uses the **Carter benchmark dataset** (`car-s-91`):
- 682 exams
- 16,925 students

## License

This project is developed for educational purposes as part of CENG482 course.
