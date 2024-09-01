import torch
from torch import nn
from torch.nn.parallel import DataParallel


def parallel_gradient_descent(
    model: nn.Module,
    objective_function: callable,
    starting_points: list[dict],
    optimizer_class: torch.optim.Optimizer,
    optimizer_kwargs: dict,
    num_epochs: int = 100,
):
    """
    Perform gradient descent from multiple starting points in parallel across multiple GPUs.

    Parameters:
    - model: A PyTorch model whose parameters are to be optimized.
    - objective_function: A function that takes the model as input and returns the scalar loss to minimize.
    - starting_points: A list of dictionaries where each dictionary represents the model state_dict for a starting point.
    - optimizer_class: The PyTorch optimizer class to be used (e.g., optim.SGD, optim.Adam).
    - optimizer_kwargs: A dictionary of keyword arguments for the optimizer.
    - num_epochs: Number of epochs to run the optimization.

    Returns:
    - best_params: The parameters of the model that achieved the lowest loss.
    - lowest_loss: The lowest loss achieved.
    """

    # Check if multiple GPUs are available
    if torch.cuda.device_count() == 0:
        raise Exception(
            "No GPU found, please make sure you have GPUs available."
        )

    # Distribute model to all available GPUs
    model = DataParallel(model).cuda()

    lowest_loss = float("inf")
    best_params = None

    # Divide the starting points across available GPUs
    starting_points_per_gpu = len(starting_points) // torch.cuda.device_count()

    # Process each batch of starting points in parallel across GPUs
    for i in range(0, len(starting_points), starting_points_per_gpu):
        batch = starting_points[i : i + starting_points_per_gpu]

        # Parallel processing of each starting point in the batch
        for start_point in batch:
            # Each process needs to clone the model to avoid shared state
            local_model = nn.DataParallel(model.module.__class__().cuda())
            local_model.load_state_dict(start_point)

            optimizer = optimizer_class(
                local_model.parameters(), **optimizer_kwargs
            )

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                loss = objective_function(local_model)
                loss.backward()
                optimizer.step()

                # Update the best parameters and lowest loss
                with torch.no_grad():
                    if loss.item() < lowest_loss:
                        lowest_loss = loss.item()
                        best_params = {
                            name: param.clone().cpu()
                            for name, param in local_model.module.named_parameters()
                        }

    # Load the best parameters found into the original model
    model.module.load_state_dict(best_params)

    return best_params, lowest_loss


# Note: You should define the model, objective_function, optimizer_class, and optimizer_kwargs according to your specific problem.
# Example usage:
# model = YourModel()
# starting_points = [model.state_dict() for _ in range(number_of_starting_points)]
# optimizer_class = optim.Adam
# optimizer_kwargs = {'lr': 0.001}
# best_params, lowest_loss = parallel_gradient_descent(model, objective_function, starting_points, optimizer_class, optimizer_kwargs)
