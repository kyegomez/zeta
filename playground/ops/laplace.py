import matplotlib.pyplot as plt
from zeta.ops.laplace import laplace_solver, follow_gradient

# Define the mesh size and the start and end points
mesh_size = 50
start = (0, 0)
end = (mesh_size - 1, mesh_size - 1)

# Solve the Laplace equation
mesh = laplace_solver(mesh_size, start, end)

# Follow the gradient to find the path
path = follow_gradient(mesh, start)

# Convert the path to a format suitable for plotting
path_x, path_y = zip(*path)

# Plot the mesh and the path
plt.figure(figsize=(8, 8))
plt.imshow(mesh, cmap="hot", interpolation="nearest")
plt.plot(path_y, path_x, color="cyan")  # Note the reversal of x and y
plt.colorbar(label="Potential")
plt.title("Solution to Laplace Equation and Path")
plt.show()
