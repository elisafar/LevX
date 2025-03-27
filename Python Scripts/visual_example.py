import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the transformation matrix
A = np.array([[2, 1],
              [1, 2]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvectors = eigenvectors / np.abs(eigenvectors).max(axis=0)  # Normalize

# Generate vectors
vector1 = 0.5 * eigenvectors[:, 0]  # Scaled eigenvector 1
vector2 = -0.5 * eigenvectors[:, 1]  # Scaled eigenvector 2
vector3 = np.array([-0.55, 1.11])  # Not aligned with eigenvectors

# Define starting points
start_point1 = np.array([1.11, 0.66])
start_point2 = np.array([0.66, -0.66])
start_point3 = np.array([0.22, 0.66])

# Generate grid of points
x_vals = np.linspace(-2, 2, 10)
y_vals = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x_vals, y_vals)
grid_points = np.vstack([X.ravel(), Y.ravel()])


# Transform grid points
transformed_points = A @ grid_points

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.grid(True, linestyle=":", linewidth=0.5)

# Scatter plot for points
points = ax.scatter(grid_points[0], grid_points[1], color='red', s=2, label="Original Points")

# Plot eigenvectors
colors = ['purple', 'blue']
for i in range(len(eigenvalues)):
    ev = eigenvectors[:, i] * 9  # Scale for visibility
    ax.arrow(0, 0, ev[0], ev[1], color=colors[i], width=0.005, head_width=0.2, length_includes_head=True, label=f"Eigenvector {i+1}")

# Initialize vectors and arrows
vectors = [vector1, vector2, vector3]
start_points = [start_point1, start_point2, start_point3]
vector_colors = ['purple', 'blue', 'red']

# Create lines and arrowheads
lines = []
arrows = []
for i, (vec, start, color) in enumerate(zip(vectors, start_points, vector_colors)):
    line, = ax.plot([], [], color=color, lw=2, label=f"Vector {i+1}")
    arrow = ax.arrow(start[0], start[1], vec[0], vec[1], color=color, width=0.01, head_width=0.15, length_includes_head=True)
    lines.append(line)
    arrows.append(arrow)

# Animation function
def update(frame):
    alpha = frame / 30  # Gradual transformation factor
    
    # Update the grid points
    intermediate_points = (1 - alpha) * grid_points + alpha * transformed_points
    points.set_offsets(intermediate_points.T)  # Move points

    # Update the vectors (lines + arrows)
    for i, (line, arrow, start, vec) in enumerate(zip(lines, arrows, start_points, vectors)):
        new_start = (1 - alpha) * start + alpha * (A @ start)  # Move starting point
        new_vec = (1 - alpha) * vec + alpha * (A @ vec)  # Transform vector 

        # Update line
        line.set_data([new_start[0], new_start[0] + new_vec[0]], 
                      [new_start[1], new_start[1] + new_vec[1]])

        # Update arrowhead
        arrow.remove()  # Remove old arrow to replace it
        arrows[i] = ax.arrow(new_start[0], new_start[1], new_vec[0], new_vec[1], 
                             color=vector_colors[i], width=0.01, head_width=0.15, length_includes_head=True)

    return points, *lines, *arrows

# Create animation
ani = animation.FuncAnimation(fig, update, frames=30, interval=100, blit=False)

plt.title("Point and Vector Transformation Animation")
plt.legend()
plt.show()
