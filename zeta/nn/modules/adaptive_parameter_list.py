import torch.nn as nn


class AdaptiveParameterList(nn.ParameterList):
    """
    A container that allows for parameters to adapt their values
    based on the learning process

    Example:
        ```
        def adaptation_function(param):
            return param * 0.9

        adaptive_params = AdaptiveParameterList(
            [nn.Parameter(torch.radnn(10, 10))]
        )
        adaptive_params.adapt(adaptation_func)
        ````
    """

    def __init__(self, parameters=None):
        super(AdaptiveParameterList, self).__init__(parameters)

    def adapt(self, adaptation_functions):
        """
        adapt the parameters using the provided func

        Args:
            adaptatio_function (callable) the function to adapt the parameters
        """
        if not isinstance(adaptation_functions, dict):
            raise ValueError("adaptation_functions must be a dictionary")

        for i, param in enumerate(self):
            if i in adaptation_functions:
                adaptation_function = adaptation_functions[i]
                if not callable(adaptation_function):
                    raise ValueError("adaptation_function must be callable")
                new_param = adaptation_function(param)
                if not new_param.shape == param.shape:
                    raise ValueError(
                        "adaptation_function must return a tensor of the same"
                        " shape as the input parameter"
                    )
                self[i] = nn.Parameter(new_param)
