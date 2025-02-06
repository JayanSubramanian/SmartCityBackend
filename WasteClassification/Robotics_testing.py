import numpy as np
from scipy.optimize import minimize
import random

# SCARA Robot Parameters
L1 = 0.4  # Length of first arm (meters)
L2 = 0.3  # Length of second arm (meters)
target_positions = [
    np.array([0.3, 0.2]), np.array([0.4, 0.2]), np.array([0.5, 0.3]),
    np.array([0.4, 0.4]), np.array([0.3, 0.5]), np.array([0.2, 0.5]),
    np.array([0.1, 0.4]), np.array([0.2, 0.3])
]

# Forward Kinematics
def forward_kinematics(theta1, theta2):
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

# Jacobian Matrix
def jacobian(theta1, theta2):
    return np.array([
        [-L1 * np.sin(theta1) - L2 * np.sin(theta1 + theta2), -L2 * np.sin(theta1 + theta2)],
        [L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2), L2 * np.cos(theta1 + theta2)]
    ])

# Transformation Matrix
def transformation_matrix(theta1, theta2):
    T1 = np.array([
        [np.cos(theta1), -np.sin(theta1), 0, L1 * np.cos(theta1)],
        [np.sin(theta1), np.cos(theta1), 0, L1 * np.sin(theta1)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    T2 = np.array([
        [np.cos(theta2), -np.sin(theta2), 0, L2 * np.cos(theta2)],
        [np.sin(theta2), np.cos(theta2), 0, L2 * np.sin(theta2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Overall transformation is the product of the two matrices
    T = np.dot(T1, T2)
    return T

# Cost Function
def cost_function(angles, target):
    theta1, theta2 = angles
    end_effector = forward_kinematics(theta1, theta2)
    path_cost = np.linalg.norm(end_effector - target)
    J = jacobian(theta1, theta2)
    singularity_cost = 1.0 / (np.abs(np.linalg.det(J)) + 1e-5)
    angle_cost = np.abs(theta1) + np.abs(theta2)
    return path_cost + 10 * singularity_cost + 0.1 * angle_cost

#Genetic Algorithm for Trajectory Optimization
def genetic_algorithm(target, population_size=50, generations=100, mutation_rate=0.1):
    population = np.random.uniform(-np.pi, np.pi, (population_size, 2))
    for generation in range(generations):
        fitness_scores = np.array([cost_function(ind, target) for ind in population])
        parents = population[np.argsort(fitness_scores)[:population_size // 2]]
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.choice(parents), random.choice(parents)
            crossover_point = random.randint(0, 1)
            child = np.copy(parent1)
            child[crossover_point:] = parent2[crossover_point:]
            offspring.append(child)
        for child in offspring:
            if random.random() < mutation_rate:
                child += np.random.normal(0, 0.1, size=child.shape)
        population = np.concatenate((parents, np.array(offspring)), axis=0)
    return population[np.argmin([cost_function(ind, target) for ind in population])]

# Pick-and-Place Simulation
def pick_and_place_with_trajectory():
    target_class = random.randint(0, 7)
    target = target_positions[target_class]
    print(f"Target Position for Class {target_class}: {target}")
    optimal_angles = genetic_algorithm(target)
    print(f"Optimal Joint Angles: θ1 = {np.degrees(optimal_angles[0]):.2f}°, θ2 = {np.degrees(optimal_angles[1]):.2f}°")
    
    # Calculate and print transformation matrix
    T = transformation_matrix(optimal_angles[0], optimal_angles[1])
    print("Transformation Matrix:")
    print(T)
    
    print("Pick-and-Place operation complete.")


# Identify Singularities
def check_singularities():
    singular_points = []
    for target in target_positions:
        optimal_angles = genetic_algorithm(target)
        theta1, theta2 = optimal_angles
        J = jacobian(theta1, theta2)
        det_J = np.linalg.det(J)
        
        if abs(det_J) < 1e-5:  # Near-zero determinant means singularity
            singular_points.append((np.degrees(theta1), np.degrees(theta2), det_J, target))

    if singular_points:
        print("\nSingular Configurations Found:")
        for theta1, theta2, det_J, target in singular_points:
            print(f"θ1 = {theta1:.2f}°, θ2 = {theta2:.2f}° | det(J) = {det_J:.5f} | Target: {target}")
    else:
        print("\nNo Singular Configurations Detected.")

def evaluate_cost_efficiency(target):
    # Generate random joint angles for comparison
    random_angles = np.random.uniform(-np.pi, np.pi, (10, 2))
    random_costs = [cost_function(angles, target) for angles in random_angles]
    
    # Get the optimal angles
    optimal_angles = genetic_algorithm(target)
    optimal_cost = cost_function(optimal_angles, target)
    
    print(f"\nTarget Position: {target}")
    print(f"Optimal Angles: θ1 = {np.degrees(optimal_angles[0]):.2f}°, θ2 = {np.degrees(optimal_angles[1]):.2f}°")
    print(f"Optimal Cost: {optimal_cost:.5f}")
    print(f"Random Costs: {random_costs}")
    
    if optimal_cost < min(random_costs):
        print("✅ The optimized angles have a lower cost than random angles, proving they are cost-efficient.")
    else:
        print("❌ The optimized angles do not outperform random angles, check optimization parameters.")

# Evaluate for a random target
if __name__ == "__main__":
    random_target = random.choice(target_positions)
    evaluate_cost_efficiency(random_target)
    pick_and_place_with_trajectory()
    # Run the evaluation
    check_singularities()


