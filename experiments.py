import time
import numpy as np
import itertools
from metaheuristic import Metaheuristic

# CONFIGURATION
PROBLEM_INSTANCES = [
    "instances/instance_n50_k2_1.json",
    "instances/instance_n50_k5_4.json",
    "instances/instance_n100_k10_7.json"
]

TIME_LIMIT = 5                
N_RUNS = 2                   
RANDOM_SEED = 42            

PARAM_GRID = {
    "pop_size": [100, 150],       
    "mutation_rate": [0.3, 0.4, 0.5], 
    "elitism_rate": [0.05, 0.1]   
}

# UTILITY FUNCTIONS
def generate_param_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def run_single_experiment(problem_path, params, seed):
    # Set seed for both numpy and random for full reproducibility
    np.random.seed(seed)
    import random
    random.seed(seed)

    ga = Metaheuristic(
        time_deadline=TIME_LIMIT,
        problem_path=problem_path,
        **params
    )
    ga.run()
    
    # Greife direkt auf die gespeicherten Attribute zu
    return ga.best_fitness, TIME_LIMIT 

# MAIN EXPERIMENT LOOP
def main():
    results = []
    param_combinations = list(generate_param_combinations(PARAM_GRID))

    print(f"Total configurations: {len(param_combinations)}")
    print(f"Runs per configuration: {N_RUNS}")
    print("=" * 50)

    for problem in PROBLEM_INSTANCES:
        print(f"\nProblem: {problem}")
        print("-" * 50)
        
        problem_best_cfg = None
        problem_max_fit = -float('inf')

        for params in param_combinations:
            fitness_values = []

            for run in range(N_RUNS):
                seed = RANDOM_SEED + run
                fitness, _ = run_single_experiment(problem, params, seed)
                fitness_values.append(fitness)

            avg_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            if avg_fitness > problem_max_fit:
                problem_max_fit = avg_fitness
                problem_best_cfg = params

            print(
                f"Params={params} | "
                f"AvgFit={avg_fitness:.4f} | "
                f"Std={std_fitness:.4f}"
            )
        
        print(f"\n>> Best for {problem}: {problem_best_cfg} (Fit: {problem_max_fit:.4f})")

    print("\n" + "="*50)
    print("EXPERIMENTS COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()