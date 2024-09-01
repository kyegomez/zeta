import copy

import torch
from torch import Tensor, nn

from zeta.utils.enforce_types import enforce_types


class SLERPModelMerger(nn.Module):
    """
    A class to merge models using Spherical Linear Interpolation (SLERP).

    SLERP provides a method to interpolate between two sets of weights, which can be
    beneficial for combining models trained in different phases.

    Attributes:
        model1 (nn.Module): The first model to be merged.
        model2 (nn.Module): The second model to be merged.
        t (float): The interpolation parameter ranging from 0 (model1) to 1 (model2).

    Examples::
    model1 = nn.Linear(10, 10)
    model2 = nn.Linear(10, 10)
    model3 = nn.Linear(10, 10)
    model4 = nn.Linear(10, 10)

    merge = SLERPModelMerger(model1, model2, 0.5)
    mergedim = merge.merge()
    print(mergedim.state_dict())
    """

    @enforce_types
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        t: float = 0.5,
    ):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.t = t

    def merge(self) -> nn.Module:
        """
        Merges the models using SLERP.

        Returns:
            nn.Module: A new model with merged weights.
        """
        mergedim = self._copy_model_structure(self.model1)

        # Get the state dicts of both models
        state_dict1 = self.model1.state_dict()
        state_dict2 = self.model2.state_dict()

        # Init a state dict for the merged model
        merged_state_dict = mergedim.state_dict()

        for key in merged_state_dict.keys():
            # Perform WELP for each parameter
            w1 = state_dict1[key]
            w2 = state_dict2[key]
            merged_state_dict[key] = self._slerp(w1, w2, self.t)

        # Load the mergd state dict into the new model
        mergedim.load_state_dict(merged_state_dict)
        return mergedim

    @staticmethod
    @enforce_types
    def _slerp(w1: Tensor, w2: Tensor, t: float) -> Tensor:
        """
        Performs Spherical Linear Interpolation (SLERP) between two tensors.

        Args:
            w1 (torch.Tensor): The first tensor.
            w2 (torch.Tensor): The second tensor.
            t (float): The interpolation parameter.

        Returns:
            torch.Tensor: The interpolated tensor.
        """
        omega = torch.acos(
            torch.clamp(
                torch.dot(w1.view(-1), w2.view(-1))
                / (torch.norm(w1) * torch.norm(w2)),
                -1,
                1,
            )
        )
        sin_omega = torch.sin(omega)
        return (torch.sin((1.0 - t) * omega) / sin_omega) * w1 + (
            torch.sin(t * omega) / sin_omega
        ) * w2

    @staticmethod
    @enforce_types
    def _copy_model_structure(model: nn.Module) -> nn.Module:
        """
        Creates a new instance of a model with the same structure as the given model.

        Args:
            model (nn.Module): The model whose structure is to be copied.

        Returns:
            nn.Module: A new model with the same structure.
        """
        assert isinstance(
            model, nn.Module
        ), "model must be an nn.Module instance"
        model_copy = copy.deepcopy(model)
        return model_copy


# model1 = nn.Linear(10, 10)
# model2 = nn.Linear(10, 10)
# model3 = nn.Linear(10, 10)
# model4 = nn.Linear(10, 10)

# merge = SLERPModelMerger(model1, model2, 0.5)
# mergedim = merge.merge()
# print(mergedim.state_dict())
