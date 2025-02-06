import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define SCARA Robot parameters
L1, L2 = 0.6, 0.8  # Link lengths

def forward_kinematics(theta):
    """Compute the end-effector position given joint angles."""
    x = L1 * np.cos(theta[0]) + L2 * np.cos(theta[0] + theta[1])
    y = L1 * np.sin(theta[0]) + L2 * np.sin(theta[0] + theta[1])
    return np.array([x, y])

def cost_function(theta, target):
    """Compute Euclidean distance from target position."""
    pos = forward_kinematics(theta)
    return np.linalg.norm(pos - target)

def jacobian(theta):
    """Compute the Jacobian matrix."""
    J = np.array([[-L1*np.sin(theta[0]) - L2*np.sin(theta[0] + theta[1]), -L2*np.sin(theta[0] + theta[1])],
                  [L1*np.cos(theta[0]) + L2*np.cos(theta[0] + theta[1]), L2*np.cos(theta[0] + theta[1])]])
    return J

def condition_number(theta):
    """Compute the condition number of the Jacobian, lower is better for energy efficiency."""
    J = jacobian(theta)
    return np.linalg.cond(J)

def genetic_algorithm(target, population_size=50, generations=100, mutation_rate=0.2):
    """Genetic Algorithm for inverse kinematics."""
    population = np.random.uniform(-np.pi, np.pi, (population_size, 2))
    best_fitness = []
    
    for gen in range(generations):
        fitness = np.linalg.norm(np.apply_along_axis(forward_kinematics, 1, population) - target, axis=1)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        best_fitness.append(fitness[0])
        
        new_population = population[:population_size // 2]
        offspring = new_population + np.random.uniform(-0.1, 0.1, new_population.shape)
        population = np.vstack((new_population, offspring))
    
    best_solution = population[0]
    return best_solution, best_fitness

def hybrid_optimization(target, initial_solution):
    """Use L-BFGS-B for fine-tuning the best GA result."""
    result = minimize(cost_function, initial_solution, args=(target,), method='L-BFGS-B', bounds=[(-np.pi, np.pi), (-np.pi, np.pi)])
    return result.x

# Generate random angles for comparison
num_random_tests = 10
random_thetas = np.random.uniform(-np.pi, np.pi, (num_random_tests, 2))

# Target Position
target_position = np.array([1.0, 0.5])

# Run Genetic Algorithm
best_theta, fitness_history = genetic_algorithm(target_position)

# Optimize further using L-BFGS-B
optimized_theta = hybrid_optimization(target_position, best_theta)

# Compute cost function values
random_costs = [cost_function(theta, target_position) for theta in random_thetas]
ga_cost = cost_function(best_theta, target_position)
optimized_cost = cost_function(optimized_theta, target_position)

# Compute energy efficiency (Jacobian condition number)
random_condition_numbers = [condition_number(theta) for theta in random_thetas]
ga_condition_number = condition_number(best_theta)
optimized_condition_number = condition_number(optimized_theta)

# Results
results = {
    "Random Avg Cost": np.mean(random_costs),
    "GA Cost": ga_cost,
    "Optimized Cost": optimized_cost,
    "Random Avg Condition Number": np.mean(random_condition_numbers),
    "GA Condition Number": ga_condition_number,
    "Optimized Condition Number": optimized_condition_number
}

print(results)
