"""
This file contains an example of how you can run your metaheuristic with a bound of n seconds and providing a specific problem instance. Your final submitted algorithm should work with this file
with no issue. Otherwise, it may not be able to participate in the tournament.

Example of how to call this file from the terminal:
python tester.py -d 60 -i instance01.txt

OR

python tester.py --deadline 60 --instance instance01.txt
"""

import click # May need to pip install click
from metaheuristic import Metaheuristic
from func_timeout import func_timeout, FunctionTimedOut # Requires pip install func_timeout
import time
import numpy as np

@click.command()
@click.option(
    "-d",
    "--deadline",
    type=int,
    default=60,
    help="Execution deadline"
)
@click.option(
    "-i",
    "--instance",
    type=click.Path(exists=True),
    required=True,
    help="Path to the problem instance to be solved"
)
def run_metaheuristic(deadline, instance):
    # Initialize the metaheuristic with the chosen instance and deadline
    met = Metaheuristic(deadline, instance)
    total_time = None
    
    print(f"--- Starting Execution: {instance} ---")
    
    try:
        t1 = time.time()
        # Run the GA until it finishes or hits the deadline
        func_timeout(deadline, met.run)
        total_time = time.time() - t1
    except FunctionTimedOut:
        total_time = deadline
        print(f"\n[Deadline reached: {deadline}s]")

    best_fitness = met.best_fitness 
    best_sol = met.best_solution

    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Problem Instance: {instance}")
    print(f"Execution Time:   {total_time:.2f} seconds")
    
    if best_fitness is not None and best_fitness > 0:
        print(f"Best Fitness:     {best_fitness:.6f}")
        # Show a snippet of the weights (first 5 assets)
        # Note: best_sol might be a list or numpy array
        weights_snippet = [round(float(w), 4) for w in best_sol[:5]]
        print(f"Weights (First 5): {weights_snippet}...")
    else:
        print("Best Fitness:     No valid solution found (Return constraint not met).")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_metaheuristic()