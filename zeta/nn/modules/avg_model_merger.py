import copy
from typing import List

from torch import nn


class AverageModelMerger:
    """
    A class to merge multiple models by averaging their weights.

    This is a simple yet effective method to combine models trained in different stages
    (like instruction and alignment tuning) to potentially boost performance.

    Attributes:
        models (List[nn.Module]): A list of PyTorch models to be merged.

    Examples::
    # Example usage:
    model1 = nn.Linear(in_features=10, out_features=10)
    model2 = nn.Linear(in_features=10, out_features=10)
    model3 = nn.Linear(in_features=10, out_features=10)
    merge = AverageModelMerger([model1, model2, model3])
    merged_model = merge.merge_models()
    print(merged_model)
    """

    def __init__(self, models: List[nn.Module]):
        """
        Initializes the AverageModelMerger with a list of models.

        Args:
            models (List[nn.Module]): Models to be merged.
        """
        assert isinstance(models, list), "models must be a list"
        assert all(
            isinstance(model, nn.Module) for model in models
        ), "models must contain nn.Module instances"
        self.models = models

    def merge_models(self) -> nn.Module:
        """
        Merges the models by averaging their weights.

        Returns:
            nn.Module: A new model with averaged weights.
        """
        assert len(self.models) > 0, "models list must not be empty"

        merged_model = self._copy_model_structure(self.models[0])

        # Initialize a state_dict for the merged model
        merged_state_dict = merged_model.state_dict()

        # Iterate over each parameter in the model's state_dict
        for key in merged_state_dict.keys():
            # Average the corresponding parameters from each model
            merged_state_dict[key] = sum(
                model.state_dict()[key] for model in self.models
            ) / len(self.models)

        # Load the averaged state_dict into the merged model
        merged_model.load_state_dict(merged_state_dict)
        return merged_model

    @staticmethod
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


# # Example usage:

# model1 = nn.Linear(in_features=10, out_features=10)
# model2 = nn.Linear(in_features=10, out_features=10)
# model3 = nn.Linear(in_features=10, out_features=10)
# merge = AverageModelMerger([model1, model2, model3])
# merged_model = merge.merge_models()
# print(merged_model)
