import torch


def laplace_solver(mesh_size, start, end, max_iter=5000):
    """init laplace solver"""
    # Initialize the mesh with zeros
    mesh = torch.zeros(mesh_size, mesh_size)

    # Set Dirichlet BCs
    mesh[start] = 0
    mesh[end] = 1

    # Iteratively solve for the mesh values
    for _ in range(max_iter):
        mesh_new = mesh.clone()
        for i in range(1, mesh_size - 1):
            for j in range(1, mesh_size - 1):
                # Apply the Laplace operator
                mesh_new[i, j] = 0.25 * (
                    mesh[i + 1, j]
                    + mesh[i - 1, j]
                    + mesh[i, j + 1]
                    + mesh[i, j - 1]
                )

        # Update the mesh
        mesh = mesh_new

    return mesh


def follow_gradient(mesh, start):
    """Follow Gradient"""
    path = [start]
    current = start
    while mesh[current] < 1:
        # Find the neighboring cell with the highest value
        neighbors = [
            (current[0] + i, current[1] + j)
            for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        next_cell = max(neighbors, key=lambda x: mesh[x])

        # Add the next cell to the path
        path.append(next_cell)
        current = next_cell

    return path
