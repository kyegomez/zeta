import torch
import torch.nn as nn
import plotly.graph_objects as go
import numpy as np


# Helper function to convert Cartesian coordinates to spherical coordinates
def spherical_coordinates(num_points, radius):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_y = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_y * radius
        z = np.sin(theta) * radius_y * radius
        points.append([x, y * radius, z])

    return np.array(points)


# Dynamic extraction of model layers and parameters
def extract_model_layers(model):
    layers = []
    params_per_layer = []

    def recursive_layer_extraction(layer, layer_name=""):
        if isinstance(layer, nn.Module):
            num_params = sum(
                p.numel() for p in layer.parameters() if p.requires_grad
            )
            if num_params > 0:
                layers.append(
                    {
                        "name": layer_name,
                        "type": layer.__class__.__name__,
                        "num_params": num_params,
                    }
                )
                params_per_layer.append(num_params)
            # Traverse through children layers
            for name, child in layer.named_children():
                layer_full_name = f"{layer_name}.{name}" if layer_name else name
                recursive_layer_extraction(child, layer_full_name)

    recursive_layer_extraction(model)
    return layers, params_per_layer


# Function to create a dynamic 3D visualization of the model
def visualize_model_as_ball(model, max_points_per_layer=100):
    layers, params_per_layer = extract_model_layers(model)

    # Visualize each layer as a concentric sphere with points
    fig = go.Figure()
    max_radius = 10  # Maximum radius for the outer layer
    radius_step = max_radius / len(layers)

    # Create spheres and add points to each sphere
    for i, layer in enumerate(layers):
        radius = radius_step * (i + 1)
        num_params = min(
            params_per_layer[i], max_points_per_layer
        )  # Limit points per layer for clarity
        points = spherical_coordinates(num_params, radius)

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=np.linspace(0, 1, num_params),
                    colorscale="Viridis",
                    opacity=0.8,
                ),
                name=f'Layer {i + 1}: {layer["type"]}',
                hovertext=[
                    f'{layer["name"]}, Param {j + 1}' for j in range(num_params)
                ],
            )
        )

    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
            zaxis=dict(title="Z", showgrid=False, zeroline=False),
        ),
        title="Dynamic Model Visualization as Ball",
        showlegend=True,
    )

    fig.show()


# Example usage with a pretrained model
if __name__ == "__main__":
    # Load a pretrained model (e.g., ResNet18 from torchvision)
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", pretrained=True
    )

    # Visualize the model
    visualize_model_as_ball(model)
