import numpy as np
import matplotlib.pyplot as plt

# Transition probability matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])

# Initial state (e.g., sunny)
state = 0  # 0: Sunny, 1: Cloudy, 2: Rainy
states = ['Sunny', 'Cloudy', 'Rainy']

# Simulation
num_days = 1000
history = []

for _ in range(num_days):
    history.append(state)
    state = np.random.choice([0, 1, 2], p=P[state])

# Compute the steady-state distribution using eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(P.T)  # Transpose to find steady-state
steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])  # Take eigenvector for eigenvalue=1
steady_state = steady_state / np.sum(steady_state)  # Normalize

# Visualization
plt.figure(figsize=(10, 5))
plt.hist(history, bins=np.arange(4)-0.5, density=True, alpha=0.6, color='b', label="Simulation")
plt.bar(range(3), steady_state.flatten(), alpha=0.6, color='r', label="Eigenvector (Steady-State)")
plt.xticks(range(3), states)
plt.ylabel("Probability")
plt.title("Weather Simulation & Steady-State Distribution")
plt.legend()
plt.show()

# Print steady-state distribution
print(f"Steady-State Probabilities: {steady_state.flatten()}")
