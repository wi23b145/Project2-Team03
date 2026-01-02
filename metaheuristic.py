import json
import numpy as np
import time
import random

class Metaheuristic:
    """
    Genetic Algorithm for Portfolio Optimization Problem

    PROBLEM OVERVIEW:
    - We have n assets, must select exactly k of them
    - Assign weights (wᵢ) to selected assets summing to 1
    - MAXIMIZE: Diversity = Σ wᵢ * wⱼ * dᵢⱼ (for all pairs i,j)
    - CONSTRAINTS:
        * Exactly k assets selected
        * Weights sum to 1
        * Minimum return R must be met

    GENETIC ALGORITHM APPROACH:
    - Representation: Array of n real values [0,1]
    - Decoding: Select k assets with highest values, normalize weights
    - Operators: Tournament selection, crossover, mutation
    - Elitism: Keep best individuals
    """

    def read_problem_instance(self, problem_path):
        """
        Reads the problem instance from a JSON file.

        This method ONLY reads data - NO computation or search allowed here!

        Args:
            problem_path: Path to JSON file containing problem data

        The JSON format is:
        {
            "n": number of available assets,
            "k": number of assets to select,
            "r": [r₁, r₂, ..., rₙ] expected returns,
            "R": minimum required return,
            "dij": [[d₀₀, d₀₁, ...], [...], ...] distance matrix
        }
        """
        with open(problem_path, 'r') as f:
            data = json.load(f)

        # Store problem parameters
        self.n = data['n']              # Total number of assets
        self.k = data['k']              # Number of assets to select
        self.r = np.array(data['r'])    # Expected returns for each asset
        self.R = data['R']              # Minimum required portfolio return
        self.dij = np.array(data['dij']) # Distance matrix (diversity measure)

        # Epsilon: minimum weight for selected assets (linking constraint)
        self.epsilon = 0.01

    def get_best_solution(self):
        """
        Returns the best solution found so far.

        IMPORTANT: Must return a list of n real numbers representing weights.
        Format: [w₀, w₁, w₂, ..., wₙ₋₁]
        - Exactly k values should be non-zero
        - All values should sum to 1

        This method can be called DURING the run (before it finishes!)
        to check intermediate results.

        Returns:
            List of n float values (weights for each asset)
        """
        if self.best_solution is None:
            # If no solution found yet, return a random valid one
            return self._generate_random_solution()
        return self.best_solution.copy()

    def _generate_random_solution(self):
        """
        Generates a random valid solution.

        Process:
        1. Select k random assets
        2. Generate random weights for them
        3. Normalize weights to sum to 1
        4. Other assets get weight 0

        Returns:
            List of n weights
        """
        solution = np.zeros(self.n)
        # Randomly select k assets
        selected = np.random.choice(self.n, size=self.k, replace=False)
        # Generate random weights
        weights = np.random.random(self.k)
        weights = weights / weights.sum()  # Normalize to sum to 1
        solution[selected] = weights
        return solution.tolist()

    def _decode_solution(self, individual):
        """
        Decodes a chromosome into a valid portfolio solution.

        OUR REPRESENTATION TRICK:
        - Internally: chromosome = [x₀, x₁, ..., xₙ₋₁] with values in [0,1]
        - Decoding: Select the k assets with HIGHEST values
        - Then normalize their weights to sum to 1

        Example with n=5, k=2:
        chromosome = [0.3, 0.8, 0.1, 0.9, 0.2]
        → Assets 1 and 3 have highest values (0.8 and 0.9)
        → Weights: 0.8/(0.8+0.9) ≈ 0.47 and 0.9/(0.8+0.9) ≈ 0.53
        → Solution: [0, 0.47, 0, 0.53, 0]

        Args:
            individual: Chromosome (array of n real values in [0,1])

        Returns:
            Portfolio solution (array of n weights, k non-zero, sum=1)
        """
        solution = np.zeros(self.n)

        # Step 1: Find indices of k highest values
        selected_indices = np.argsort(individual)[-self.k:]

        # Step 2: Extract weights for selected assets
        selected_weights = individual[selected_indices]

        # Step 3: Normalize to sum to 1
        if selected_weights.sum() > 0:
            selected_weights = selected_weights / selected_weights.sum()
        else:
            # Edge case: if all are zero, assign equal weights
            selected_weights = np.ones(self.k) / self.k

        # Step 4: Assign weights to solution
        solution[selected_indices] = selected_weights

        return solution

    def _fitness(self, individual):
        """
        Evaluates the fitness of an individual (chromosome).

        FITNESS = DIVERSITY - PENALTIES

        Where:
        - DIVERSITY = Σᵢ Σⱼ₍ⱼ>ᵢ₎ wᵢ * wⱼ * dᵢⱼ
          (higher diversity = better portfolio)

        - PENALTIES for constraint violations:
          * Return constraint: if portfolio return < R
          * Selection constraint: if number of selected assets ≠ k

        Args:
            individual: Chromosome to evaluate

        Returns:
            Fitness value (higher is better)
        """
        # Decode chromosome to portfolio solution
        if individual.ndim == 1 and np.sum(individual > 0) == self.k:
            solution = individual  # already decoded
        else:
            solution = self._decode_solution(individual)


        # Calculate DIVERSITY (objective function)
        diversity = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                diversity += solution[i] * solution[j] * self.dij[i][j]

        # PENALTY 1: Check return constraint
        portfolio_return = np.dot(solution, self.r)
        if portfolio_return < self.R:
            return_penalty = (self.R - portfolio_return) * 1000
        else:
            return_penalty = 0

        # PENALTY 2: Check selection constraint (should be exactly k assets)
        num_selected = np.sum(solution > 0)
        selection_penalty = abs(num_selected - self.k) * 1000

        # Final fitness (higher is better)
        fitness = diversity - return_penalty - selection_penalty

        return fitness
    
    def _local_search(self, decoded_solution, steps=20, step_size=0.05):
        """
        Local search on a decoded (feasible) solution.
        Shifts small weight between selected assets to improve fitness.
        """
        best_solution = decoded_solution.copy()
        best_fitness = self._fitness(best_solution)

        selected = np.where(best_solution > 0)[0]

        for _ in range(steps):
            i, j = np.random.choice(selected, size=2, replace=False)

            new_solution = best_solution.copy()
            delta = min(step_size, new_solution[i])

            new_solution[i] -= delta
            new_solution[j] += delta

            # Re-normalize selected weights
            new_solution[selected] /= new_solution[selected].sum()

            fitness = self._fitness(new_solution)

            if fitness > best_fitness:
                best_solution = new_solution
                best_fitness = fitness

        return best_solution


    def _initialize_population(self):
        """
        Creates the initial population for the GA.

        STRATEGY: Mix random and greedy-biased individuals

        - 60% completely random: ensures diversity
        - 40% greedy-biased: favor high-return and high-diversity assets

        Why both?
        - Random: explores search space broadly
        - Greedy: starts near promising regions

        Returns:
            Population array of shape (pop_size, n)
        """
        population = []

        # PART 1: Random individuals (60%)
        num_random = int(self.pop_size * 0.6)
        for _ in range(num_random):
            individual = np.random.random(self.n)
            population.append(individual)

        # PART 2: Greedy-biased individuals (40%)
        # Calculate scores: combine returns and diversity
        avg_distances = np.mean(self.dij, axis=1)  # Average diversity per asset

        # Normalize to [0,1] range
        norm_returns = (self.r - self.r.min()) / (self.r.max() - self.r.min() + 1e-10)
        norm_distances = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-10)

        # Combined score: 50% return + 50% diversity
        scores = norm_returns * 0.5 + norm_distances * 0.5

        num_greedy = self.pop_size - num_random
        for _ in range(num_greedy):
            individual = np.random.random(self.n)
            # Bias towards high-scoring assets
            individual = individual * (1 + scores)
            population.append(individual)

        return np.array(population)

    def _tournament_selection(self, population, fitnesses, tournament_size=3):
        """
        Tournament selection: pick the best from a random subset.

        PROCESS:
        1. Randomly select tournament_size individuals
        2. Compare their fitness values
        3. Return the one with highest fitness

        Why tournament?
        - Simple and effective
        - Applies selection pressure (better individuals more likely to be selected)
        - Maintains diversity (not just selecting the absolute best)

        Args:
            population: Current population
            fitnesses: Fitness values for each individual
            tournament_size: Number of competitors (default: 3)

        Returns:
            Selected individual (copy)
        """
        # Step 1: Pick random contestants
        indices = np.random.choice(len(population), size=tournament_size, replace=False)

        # Step 2: Get their fitness values
        tournament_fitnesses = fitnesses[indices]

        # Step 3: Winner = highest fitness
        winner_idx = indices[np.argmax(tournament_fitnesses)]

        return population[winner_idx].copy()

    def _uniform_crossover(self, parent1, parent2):
        """
        Uniform crossover: each gene randomly chosen from either parent.

        PROCESS:
        For each position i:
        - 50% chance: child1[i] = parent1[i], child2[i] = parent2[i]
        - 50% chance: child1[i] = parent2[i], child2[i] = parent1[i]

        Example:
        parent1 = [0.3, 0.8, 0.2, 0.9]
        parent2 = [0.6, 0.1, 0.7, 0.4]
        mask    = [T,   F,   T,   F]
        child1  = [0.3, 0.1, 0.2, 0.4]
        child2  = [0.6, 0.8, 0.7, 0.9]

        Args:
            parent1, parent2: Parent chromosomes

        Returns:
            Two children chromosomes
        """
        mask = np.random.random(self.n) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _blend_crossover(self, parent1, parent2, alpha=0.5):
        """
        BLX-α crossover: blend genes with random values in extended range.

        PROCESS:
        For each gene i:
        1. Find min and max from both parents
        2. Extend range by α on both sides
        3. Pick random value in extended range

        Example with α=0.5:
        parent1[i] = 0.2, parent2[i] = 0.8
        range = 0.8 - 0.2 = 0.6
        extended: [0.2 - 0.5*0.6, 0.8 + 0.5*0.6] = [-0.1, 1.1]
        clipped: [0, 1]
        child[i] = random value in [0, 1]

        This creates children that can go slightly beyond parent values.

        Args:
            parent1, parent2: Parent chromosomes
            alpha: Extension parameter (default: 0.5)

        Returns:
            Two children chromosomes
        """
        child1 = np.zeros(self.n)
        child2 = np.zeros(self.n)

        for i in range(self.n):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val

            # Extend the range
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val

            # Generate children values (clipped to [0,1])
            child1[i] = np.random.uniform(max(0, lower), min(1, upper))
            child2[i] = np.random.uniform(max(0, lower), min(1, upper))

        return child1, child2

    def _gaussian_mutation(self, individual):
        """
        Gaussian mutation: add random noise to genes.

        PROCESS:
        For each gene:
        - With probability mutation_rate: add Gaussian noise N(0, 0.1)
        - Clip result to [0,1] range

        Example:
        individual = [0.3, 0.8, 0.2]
        After mutation: [0.35, 0.8, 0.28] (genes 0 and 2 mutated)

        This creates small variations, helping fine-tune solutions.

        Args:
            individual: Chromosome to mutate

        Returns:
            Mutated chromosome
        """
        mutated = individual.copy()

        # Determine which genes to mutate
        mutation_mask = np.random.random(self.n) < self.mutation_rate

        # Add Gaussian noise to selected genes
        mutated[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))

        # Keep values in [0,1] range
        mutated = np.clip(mutated, 0, 1)

        return mutated

    def _swap_mutation(self, individual):
        """
        Swap mutation: exchange values of two random genes.

        PROCESS:
        1. Pick two random positions
        2. Swap their values

        Example:
        individual = [0.3, 0.8, 0.2, 0.9]
        Swap positions 1 and 3
        Result:     [0.3, 0.9, 0.2, 0.8]

        This can dramatically change which assets are selected.

        Args:
            individual: Chromosome to mutate

        Returns:
            Mutated chromosome
        """
        mutated = individual.copy()
        idx1, idx2 = np.random.choice(self.n, size=2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated

    def run(self):
        """
        Main execution of the Genetic Algorithm.

        ALGORITHM STRUCTURE:
        1. Read problem instance
        2. Initialize population
        3. Evaluate initial population
        4. MAIN LOOP (while time available):
           a. Selection: pick parents
           b. Crossover: create offspring
           c. Mutation: modify offspring
           d. Evaluation: compute fitness
           e. Replacement: form next generation (with elitism)
           f. Update best solution
        5. Return when time is up
        """
        # STEP 1: Read problem data
        self.read_problem_instance(self.problem_path)

        start_time = time.time()

        # STEP 2: Initialize population
        population = self._initialize_population()

        # STEP 3: Evaluate initial population
        fitnesses = np.array([self._fitness(ind) for ind in population])

        # Track best solution found
        best_idx = np.argmax(fitnesses)
        self.best_solution = self._decode_solution(population[best_idx]).tolist()
        best_fitness = fitnesses[best_idx]

        generation = 0
        no_improvement_count = 0

        # STEP 4: MAIN EVOLUTIONARY LOOP
        while time.time() - start_time < self.time_deadline - 0.1:
            generation += 1

            # === ELITISM: Keep best individuals ===
            elite_size = int(self.pop_size * self.elitism_rate)
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            elite = [population[i].copy() for i in elite_indices]

            # === GENERATE OFFSPRING ===
            offspring = []

            while len(offspring) < self.pop_size - elite_size:
                # SELECTION: Pick two parents
                parent1 = self._tournament_selection(population, fitnesses, self.tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, self.tournament_size)

                # CROSSOVER: Create children
                if np.random.random() < self.crossover_rate:
                    # Randomly choose crossover type
                    if np.random.random() < 0.5:
                        child1, child2 = self._uniform_crossover(parent1, parent2)
                    else:
                        child1, child2 = self._blend_crossover(parent1, parent2)
                else:
                    # No crossover: just copy parents
                    child1, child2 = parent1.copy(), parent2.copy()

                # MUTATION: Modify children
                if np.random.random() < self.mutation_rate:
                    # Randomly choose mutation type
                    if np.random.random() < 0.5:
                        child1 = self._gaussian_mutation(child1)
                    else:
                        child1 = self._swap_mutation(child1)

                if np.random.random() < self.mutation_rate:
                    if np.random.random() < 0.5:
                        child2 = self._gaussian_mutation(child2)
                    else:
                        child2 = self._swap_mutation(child2)

                offspring.append(child1)
                if len(offspring) < self.pop_size - elite_size:
                    offspring.append(child2)

            # === REPLACEMENT: Form next generation ===
            population = np.array(elite + offspring)

            # === EVALUATION: Calculate fitness ===
            fitnesses = np.array([self._fitness(ind) for ind in population])

            # === UPDATE BEST SOLUTION ===
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness

                decoded = self._decode_solution(population[current_best_idx])
                improved = self._local_search(decoded)

                self.best_solution = improved.tolist()
                no_improvement_count = 0

            else:
                no_improvement_count += 1

            # === ADAPTIVE MUTATION ===
            # If stuck (no improvement for 20 generations), increase mutation
            if no_improvement_count > 20:
                self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
                no_improvement_count = 0

            # Early stopping if time is running out
            if time.time() - start_time > self.time_deadline * 0.95:
                break

    def __init__(self, time_deadline, problem_path,
                 pop_size=100,
                 crossover_rate=0.8,
                 mutation_rate=0.2,
                 elitism_rate=0.1,
                 tournament_size=3,
                 **kwargs):
        """
        Initializes the Genetic Algorithm with hyperparameters.

        HYPERPARAMETERS EXPLAINED:

        pop_size (default: 100)
        - Population size
        - Larger = more diversity, but slower
        - Smaller = faster, but may miss good solutions

        crossover_rate (default: 0.8)
        - Probability of applying crossover
        - High values (0.7-0.9) = more exploitation of good solutions
        - Low values = more exploration

        mutation_rate (default: 0.2)
        - Probability of mutating each child
        - High values = more exploration, may prevent convergence
        - Low values = faster convergence, may get stuck

        elitism_rate (default: 0.1)
        - Fraction of best individuals kept unchanged
        - Ensures best solutions aren't lost
        - Typical: 5-15% of population

        tournament_size (default: 3)
        - Number of individuals competing in selection
        - Larger = stronger selection pressure
        - Smaller = more diversity maintained

        Args:
            time_deadline: Maximum computation time (seconds)
            problem_path: Path to problem instance JSON file
            **kwargs: Additional optional parameters
        """
        self.problem_path = problem_path
        self.best_solution = None
        self.time_deadline = time_deadline

        # Configure hyperparameters
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size