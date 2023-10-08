import torch


def newtons_method(f, df, x0, tol=1e-5, max_iter=100):
    """
    Newton's method for finding roots of a function.

    Usage:
        f = lambda x: x**2 - 1
        df = lambda x: 2*x
        newtons_method(f, df, 0.5)
        tensor(1.0000)
    """
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        if torch.abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


# Broyden Method
def broydens_method(f, J, x0, tol=1e-5, max_iter=100):
    """
    Broyden's method for finding roots of a function.

    Usage:
        f = lambda x: x**2 - 1
        J = lambda x: 2*x
        broydens_method(f, J, 0.5)
        tensor(1.0000)

    """
    x = x0
    B = J(x)
    for _ in range(max_iter):
        s = torch.linalg.solve(-f(x), B)[0]
        x_new = x + s
        if torch.norm(x_new - x) < tol:
            return x_new
        y = f(x_new) - f(x)
        B += (y - B @ s) @ s.T / (s.T @ s)
        x = x_new
    return x
