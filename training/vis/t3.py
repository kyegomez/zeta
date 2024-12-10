import plotly.graph_objects as go
from torchvision import models
import numpy as np


def visualize_model_3d(model):
    def get_layer_info(module, depth=0, path=""):
        layers = []
        for name, child in module.named_children():
            child_path = f"{path}/{name}".lstrip("/")
            child_info = {
                "name": child_path,
                "type": child.__class__.__name__,
                "params": sum(
                    p.numel() for p in child.parameters() if p.requires_grad
                ),
                "depth": depth,
            }
            layers.append(child_info)
            layers.extend(get_layer_info(child, depth + 1, child_path))
        return layers

    layers = get_layer_info(model)

    fig = go.Figure()

    max_depth = max(layer["depth"] for layer in layers)
    max_params = max(layer["params"] for layer in layers)

    # Calculate positions on a sphere
    phi = np.linspace(0, np.pi, len(layers))
    theta = np.linspace(0, 2 * np.pi, len(layers))

    # Create nodes
    for i, layer in enumerate(layers):
        r = (layer["depth"] + 1) / (max_depth + 1)  # Radius based on depth
        x = r * np.sin(phi[i]) * np.cos(theta[i])
        y = r * np.sin(phi[i]) * np.sin(theta[i])
        z = r * np.cos(phi[i])

        size = (layer["params"] / max_params * 20 + 5) if max_params > 0 else 5

        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers",
                marker=dict(
                    size=size,
                    color=layer["depth"],
                    colorscale="Viridis",
                    opacity=0.8,
                ),
                text=f"{layer['name']}<br>{layer['type']}<br>Params: {layer['params']}",
                hoverinfo="text",
            )
        )

    # Create edges
    for i in range(1, len(layers)):
        prev_layer = layers[i - 1]
        curr_layer = layers[i]

        r_prev = (prev_layer["depth"] + 1) / (max_depth + 1)
        r_curr = (curr_layer["depth"] + 1) / (max_depth + 1)

        x_prev = r_prev * np.sin(phi[i - 1]) * np.cos(theta[i - 1])
        y_prev = r_prev * np.sin(phi[i - 1]) * np.sin(theta[i - 1])
        z_prev = r_prev * np.cos(phi[i - 1])

        x_curr = r_curr * np.sin(phi[i]) * np.cos(theta[i])
        y_curr = r_curr * np.sin(phi[i]) * np.sin(theta[i])
        z_curr = r_curr * np.cos(phi[i])

        fig.add_trace(
            go.Scatter3d(
                x=[x_prev, x_curr],
                y=[y_prev, y_curr],
                z=[z_prev, z_curr],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.5)", width=2),
                hoverinfo="none",
            )
        )

    fig.update_layout(
        title="Spherical 3D Model Architecture Visualization",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        width=900,
        height=700,
        margin=dict(r=0, l=0, b=0, t=40),
    )

    return fig


# Example usage
model = models.vgg16(pretrained=True)
fig = visualize_model_3d(model)
fig.show()
