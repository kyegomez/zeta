import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import plotly.io as pio


def visualize_parameters(
    model,
    dim="3d",
    color_by="mean",
    group_by=None,
    colorscale="Viridis",
    compare_model=None,
):
    params_info = get_params_info(model)
    if compare_model:
        params_info_2 = get_params_info(compare_model)
        return create_comparison_plot(
            params_info, params_info_2, dim, color_by, colorscale
        )

    if dim == "3d":
        fig = create_3d_plot(params_info, color_by, colorscale)
    elif dim == "2d":
        fig = create_2d_plot(params_info, color_by, colorscale)
    else:
        fig = create_heatmap(params_info, color_by)

    if group_by == "layer":
        fig = group_by_layer(fig, params_info)

    add_model_summary(fig, model, params_info)
    add_distribution_histogram(fig, params_info)

    return fig


def get_params_info(model):
    params_info = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = name.split(".")[0]
            params_info.append(
                {
                    "name": name,
                    "layer": layer_name,
                    "shape": param.shape,
                    "numel": param.numel(),
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "grad_mean": (
                        param.grad.mean().item()
                        if param.grad is not None
                        else 0
                    ),
                    "grad_std": (
                        param.grad.std().item() if param.grad is not None else 0
                    ),
                    "type": param.__class__.__name__,
                    "data": param.data.flatten().tolist(),
                }
            )
    params_info.sort(key=lambda x: x["numel"], reverse=True)
    return params_info


def create_3d_plot(params_info, color_by, colorscale):
    fig = go.Figure()

    n_params = len(params_info)
    grid_size = int(np.ceil(np.cbrt(n_params)))
    x, y, z = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    x, y, z = (
        x.flatten()[:n_params],
        y.flatten()[:n_params],
        z.flatten()[:n_params],
    )

    max_numel = max(p["numel"] for p in params_info)

    marker_shapes = {"Weight": "circle", "Bias": "square"}

    for i, param in enumerate(params_info):
        size = np.log1p(param["numel"]) / np.log1p(max_numel) * 20 + 5
        color = param[color_by]

        fig.add_trace(
            go.Scatter3d(
                x=[x[i]],
                y=[y[i]],
                z=[z[i]],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    colorscale=colorscale,
                    opacity=0.8,
                    symbol=marker_shapes.get(param["type"], "circle"),
                    colorbar=dict(title=color_by.capitalize()),
                ),
                text=create_hover_text(param),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="3D Parameter Visualization",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"
        ),
        width=900,
        height=700,
        margin=dict(r=0, l=0, b=0, t=40),
        coloraxis=dict(colorscale=colorscale),
    )

    return fig


def create_2d_plot(params_info, color_by, colorscale):
    fig = go.Figure()

    n_params = len(params_info)
    grid_size = int(np.ceil(np.sqrt(n_params)))
    x, y = np.mgrid[0:grid_size, 0:grid_size]
    x, y = x.flatten()[:n_params], y.flatten()[:n_params]

    max_numel = max(p["numel"] for p in params_info)

    marker_shapes = {"Weight": "circle", "Bias": "square"}

    for i, param in enumerate(params_info):
        size = np.log1p(param["numel"]) / np.log1p(max_numel) * 20 + 5
        color = param[color_by]

        fig.add_trace(
            go.Scatter(
                x=[x[i]],
                y=[y[i]],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    colorscale=colorscale,
                    opacity=0.8,
                    symbol=marker_shapes.get(param["type"], "circle"),
                    colorbar=dict(title=color_by.capitalize()),
                ),
                text=create_hover_text(param),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="2D Parameter Visualization",
        xaxis_title="X",
        yaxis_title="Y",
        width=900,
        height=700,
        margin=dict(r=0, l=0, b=0, t=40),
        coloraxis=dict(colorscale=colorscale),
    )

    return fig


def create_heatmap(params_info, color_by):
    data = [param[color_by] for param in params_info]
    names = [param["name"] for param in params_info]

    fig = go.Figure(
        data=go.Heatmap(
            z=[data], y=["Parameters"], x=names, colorscale="Viridis"
        )
    )

    fig.update_layout(
        title="Parameter Heatmap",
        xaxis_title="Parameter Name",
        yaxis_title="",
        width=1200,
        height=400,
    )

    return fig


def create_comparison_plot(
    params_info_1, params_info_2, dim, color_by, colorscale
):
    fig = sp.make_subplots(
        rows=1, cols=2, subplot_titles=("Model 1", "Model 2")
    )

    if dim == "3d":
        fig1 = create_3d_plot(params_info_1, color_by, colorscale)
        fig2 = create_3d_plot(params_info_2, color_by, colorscale)
    elif dim == "2d":
        fig1 = create_2d_plot(params_info_1, color_by, colorscale)
        fig2 = create_2d_plot(params_info_2, color_by, colorscale)
    else:
        fig1 = create_heatmap(params_info_1, color_by)
        fig2 = create_heatmap(params_info_2, color_by)

    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(height=600, width=1200, title_text="Model Comparison")
    return fig


def add_distribution_histogram(fig, params_info):
    data = [param["data"] for param in params_info]
    flat_data = [item for sublist in data for item in sublist]

    histogram = go.Figure(data=[go.Histogram(x=flat_data)])
    histogram.update_layout(title_text="Parameter Distribution")

    fig.add_trace(histogram.data[0])
    return fig


def add_gradient_flow(fig, model):
    # Placeholder for gradient flow visualization
    # This would require tracking gradients during backward pass
    pass


def create_hover_text(param):
    return (
        f"Name: {param['name']}<br>"
        f"Shape: {param['shape']}<br>"
        f"Elements: {param['numel']}<br>"
        f"Mean: {param['mean']:.4f}<br>"
        f"Std: {param['std']:.4f}<br>"
        f"Grad Mean: {param['grad_mean']:.4f}<br>"
        f"Grad Std: {param['grad_std']:.4f}<br>"
        f"Type: {param['type']}"
    )


def group_by_layer(fig, params_info):
    layers = sorted(set(p["layer"] for p in params_info))
    fig.update_layout(
        updatemenus=[
            {
                "buttons": (
                    [
                        {
                            "label": layer,
                            "method": "update",
                            "args": [
                                {
                                    "visible": [
                                        p["layer"] == layer for p in params_info
                                    ]
                                }
                            ],
                        }
                        for layer in layers
                    ]
                    + [
                        {
                            "label": "All",
                            "method": "update",
                            "args": [{"visible": [True] * len(params_info)}],
                        }
                    ]
                ),
                "direction": "down",
                "showactive": True,
            }
        ]
    )
    return fig


def add_model_summary(fig, model, params_info):
    total_params = sum(p["numel"] for p in params_info)
    summary = f"Total parameters: {total_params:,}<br>"
    summary += f"Layers: {len(set(p['layer'] for p in params_info))}<br>"
    summary += f"Model type: {model.__class__.__name__}"

    fig.add_annotation(
        x=0.95,
        y=0.95,
        xref="paper",
        yref="paper",
        text=summary,
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
    )
    return fig


def export_html(fig, filename="parameter_visualization.html"):
    pio.write_html(fig, file=filename, auto_open=True)


# Example usage
if __name__ == "__main__":
    from torchvision import models

    model = models.resnet18(pretrained=True)
    fig = visualize_parameters(
        model, dim="3d", color_by="mean", group_by="layer", colorscale="Plasma"
    )

    # Uncomment to compare two models
    # model2 = models.resnet34(pretrained=True)
    # fig = visualize_parameters(model, dim='3d', color_by='mean', colorscale='Plasma', compare_model=model2)

    fig.show()
    export_html(fig)
