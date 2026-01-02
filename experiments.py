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

TIME_LIMIT = 20          # seconds per run
N_RUNS = 3               # repetitions per configuration
RANDOM_SEED = 42         # for reproducibility


# Hyperparameter grid (keep small & meaningful)
PARAM_GRID = {
    "pop_size": [80, 100],
    "mutation_rate": [0.1, 0.2],
    "elitism_rate": [0.05, 0.1]
}


# UTILITY FUNCTIONS

def generate_param_combinations(grid):
    """Generate all combinations of hyperparameters."""
    keys = grid.keys()
    values = grid.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def run_single_experiment(problem_path, params, seed):
    """Run one GA execution and return best fitness + runtime."""
    np.random.seed(seed)

    start_time = time.time()
    ga = Metaheuristic(
        time_deadline=TIME_LIMIT,
        problem_path=problem_path,
        **params
    )
    ga.run()
    runtime = time.time() - start_time

    best_solution = np.array(ga.get_best_solution())
    best_fitness = ga._fitness(best_solution)

    return best_fitness, runtime


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

        for params in param_combinations:
            fitness_values = []
            runtimes = []

            for run in range(N_RUNS):
                seed = RANDOM_SEED + run
                fitness, runtime = run_single_experiment(problem, params, seed)
                fitness_values.append(fitness)
                runtimes.append(runtime)

            avg_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)
            avg_runtime = np.mean(runtimes)

            result = {
                "problem": problem,
                "params": params,
                "avg_fitness": avg_fitness,
                "std_fitness": std_fitness,
                "avg_runtime": avg_runtime
            }

            results.append(result)

            print(
                f"Params={params} | "
                f"AvgFit={avg_fitness:.4f} | "
                f"Std={std_fitness:.4f} | "
                f"Time={avg_runtime:.2f}s"
            )

    print("\n=== SUMMARY ===")
    best = max(results, key=lambda x: x["avg_fitness"])
    print("Best configuration:")
    print(best)


if __name__ == "__main__":
    main()
