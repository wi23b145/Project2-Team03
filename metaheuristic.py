import json
import numpy as np
import time
import random

class Metaheuristic:
    def __init__(self, time_deadline, problem_path, 
                 pop_size=150, crossover_rate=0.8, mutation_rate=0.4, 
                 elitism_rate=0.1, tournament_size=3, **kwargs):
        """
        Initializes the Genetic Algorithm with chosen hyperparameters.
        """
        self.problem_path = problem_path
        self.time_deadline = time_deadline
        self.best_solution = None
        self.best_fitness = -float('inf')
        
        # Hyperparameters
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size

    def read_problem_instance(self, problem_path):
        """
        Loads the problem data from a JSON file.
        """
        with open(problem_path, 'r') as f:
            data = json.load(f)

        self.n = data['n']
        self.k = data['k']
        self.r = np.array(data['r'])
        self.R = data['R']
        self.dij = np.array(data['dij'])

    def get_best_solution(self):
        """
        Returns the best valid solution found during the run.
        """
        if self.best_solution is None:
            return self._generate_random_solution()
        return self.best_solution

    def _generate_random_solution(self):
        """
        Fallback method to generate a simple random valid solution.
        """
        solution = np.zeros(self.n)
        selected = np.random.choice(self.n, size=self.k, replace=False)
        weights = np.random.random(self.k)
        weights /= weights.sum()
        solution[selected] = weights
        return solution.tolist()

    def _decode_solution(self, individual):
        """
        Decodes a continuous chromosome into a valid portfolio:
        Selects top k assets and normalizes weights to sum to 1.
        """
        solution = np.zeros(self.n)
        # Select k assets with highest values in the individual
        selected_indices = np.argsort(individual)[-self.k:]
        selected_weights = individual[selected_indices]

        s_sum = selected_weights.sum()
        if s_sum > 0:
            selected_weights /= s_sum
        else:
            selected_weights = np.ones(self.k) / self.k

        solution[selected_indices] = selected_weights
        return solution

    def _calculate_diversity(self, solution):
        """
        Calculates the diversity score: 0.5 * w^T * D * w.
        Uses matrix multiplication for performance.
        """
        return 0.5 * np.dot(solution.T, np.dot(self.dij, solution))

    def _fitness(self, individual):
        """
        Evaluates the fitness. Returns -inf if return constraint is not met.
        """
        solution = self._decode_solution(individual)
        portfolio_return = np.dot(solution, self.r)
        
        # Tournament requirement: Fitness must be 0 or -inf if return R is not met
        if portfolio_return < self.R:
            return -float('inf')
        
        return self._calculate_diversity(solution)

    def _local_search(self, solution):
        """
        Exploitation: Refines the current solution by shifting weights 
        between selected assets while maintaining the return constraint.
        """
        best_sol = solution.copy()
        best_fit = self._calculate_diversity(best_sol)
        selected = np.where(best_sol > 0)[0]
        if len(selected) < 2: return best_sol

        for _ in range(30): # Number of local optimization attempts
            i, j = np.random.choice(selected, 2, replace=False)
            delta = random.uniform(0, 0.05)
            if best_sol[i] < delta: continue
            
            new_sol = best_sol.copy()
            new_sol[i] -= delta
            new_sol[j] += delta
            
            # Check if return constraint still holds
            if np.dot(new_sol, self.r) >= self.R:
                new_fit = self._calculate_diversity(new_sol)
                if new_fit > best_fit:
                    best_sol = new_sol
                    best_fit = new_fit
        return best_sol

    def _initialize_population(self):
        """
        Initializes population with a bias towards assets with high return 
        and high average distance to ensure initial feasibility.
        """
        avg_distances = np.mean(self.dij, axis=1)
        norm_r = (self.r - self.r.min()) / (self.r.max() - self.r.min() + 1e-9)
        norm_d = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-9)
        scores = 0.5 * norm_r + 0.5 * norm_d

        population = np.random.random((self.pop_size, self.n))
        # 40% biased individuals to improve convergence speed
        population[int(self.pop_size * 0.6):] *= (1 + scores)
        return population

    def _tournament_selection(self, population, fitnesses):
        """
        Selects the best individual among a random sample.
        """
        indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
        winner_idx = indices[np.argmax(fitnesses[indices])]
        return population[winner_idx]

    def run(self):
        """
        Main GA execution loop. Optimized with vectorization and adaptive mutation.
        """
        self.read_problem_instance(self.problem_path)
        start_time = time.time()
        base_mutation_rate = self.mutation_rate
        
        population = self._initialize_population()
        self.best_fitness = -float('inf')
        no_imp_counter = 0
        
        # Loop until deadline (with safety margin)
        while (time.time() - start_time) < (self.time_deadline - 0.7):
            # Calculate fitness for entire population
            fitnesses = np.array([self._fitness(ind) for ind in population])
            
            # Global best update
            current_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[current_best_idx]
            
            if gen_best_fitness > self.best_fitness:
                decoded = self._decode_solution(population[current_best_idx])
                improved = self._local_search(decoded)
                improved_fit = self._calculate_diversity(improved)
                
                if improved_fit > self.best_fitness:
                    self.best_fitness = improved_fit
                    self.best_solution = improved.tolist()
                    no_imp_counter = 0
                    self.mutation_rate = base_mutation_rate # Reset after improvement
            else:
                no_imp_counter += 1

            # Elitism: Keep top individuals
            elite_size = max(1, int(self.pop_size * self.elitism_rate))
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            next_gen = [population[i].copy() for i in elite_indices]

            # Reproduction
            while len(next_gen) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                
                # Vectorized BLX-alpha Crossover
                if random.random() < self.crossover_rate:
                    alpha = 0.5
                    diff = np.abs(p1 - p2)
                    low = np.maximum(0, np.minimum(p1, p2) - alpha * diff)
                    high = np.minimum(1, np.maximum(p1, p2) + alpha * diff)
                    child = np.random.uniform(low, high)
                else:
                    child = p1.copy() if random.random() < 0.5 else p2.copy()
                
                # Mutation (Gaussian or Swap)
                if random.random() < self.mutation_rate:
                    if random.random() < 0.5:
                        child = np.clip(child + np.random.normal(0, 0.1, self.n), 0, 1)
                    else:
                        idx = np.random.choice(self.n, 2, replace=False)
                        child[idx[0]], child[idx[1]] = child[idx[1]], child[idx[0]]
                
                next_gen.append(child)
            
            population = np.array(next_gen)
            
            # Adaptive Mutation: Increase exploration if stuck in local optimum
            if no_imp_counter > 25:
                self.mutation_rate = min(0.6, self.mutation_rate * 1.05)
                no_imp_counter = 0