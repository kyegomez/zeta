import plotly.graph_objects as go
import numpy as np
from transformers import BertModel, BertTokenizer


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


# Function to extract transformer layers and parameters (specific for BERT here)
def extract_transformer_layers(model):
    layers = []
    params_per_layer = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params > 0:
            layers.append(
                {"name": name, "num_params": num_params, "shape": param.shape}
            )
            params_per_layer.append(num_params)

    return layers, params_per_layer


# Function to visualize the transformer model components as a 3D ball structure
def visualize_transformer_as_ball(model, max_points_per_layer=100):
    layers, params_per_layer = extract_transformer_layers(model)

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
                name=f'Layer {i + 1}: {layer["name"]}',
                hovertext=[
                    f'{layer["name"]}, Shape: {layer["shape"]}'
                    for j in range(num_params)
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
        title="Transformer Model Visualization as Ball",
        showlegend=True,
    )

    fig.show()


# Visualizing Attention Weights
def visualize_attention_weights(
    attention_weights, sequence_length, attention_heads
):
    fig = go.Figure()

    # For each attention head, draw the attention matrix as connections between tokens
    for head in range(attention_heads):
        attention = (
            attention_weights[0, head].detach().numpy()
        )  # Get attention weights for this head
        points = spherical_coordinates(
            sequence_length, radius=5 + head
        )  # Slightly different radii for heads

        # Add nodes (tokens)
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color=np.linspace(0, 1, sequence_length),
                    colorscale="Plasma",
                    opacity=0.8,
                ),
                name=f"Attention Head {head + 1}",
                hovertext=[f"Token {i + 1}" for i in range(sequence_length)],
            )
        )

        # Add edges (attention weights between tokens)
        for i in range(sequence_length):
            for j in range(sequence_length):
                if (
                    i != j and attention[i, j] > 0.1
                ):  # Only show significant attention weights
                    fig.add_trace(
                        go.Scatter3d(
                            x=[points[i, 0], points[j, 0]],
                            y=[points[i, 1], points[j, 1]],
                            z=[points[i, 2], points[j, 2]],
                            mode="lines",
                            line=dict(color="rgba(0, 0, 255, 0.2)", width=2),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", showgrid=False, zeroline=False),
            yaxis=dict(title="Y", showgrid=False, zeroline=False),
            zaxis=dict(title="Z", showgrid=False, zeroline=False),
        ),
        title="Multi-Head Attention Weights",
        showlegend=True,
    )

    fig.show()


# Example usage with a pretrained BERT model
if __name__ == "__main__":
    # Load a pretrained transformer model (e.g., BERT) from Huggingface
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Example sentence
    sentence = "Transformers are powerful models for NLP tasks."
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)

    # Visualize the model structure
    visualize_transformer_as_ball(model)

    # Visualize attention weights (BERT has 12 attention heads by default)
    attention_weights = outputs.attentions[
        -1
    ]  # Get the attention weights from the last layer
    sequence_length = inputs["input_ids"].shape[1]
    visualize_attention_weights(
        attention_weights, sequence_length, attention_heads=12
    )
