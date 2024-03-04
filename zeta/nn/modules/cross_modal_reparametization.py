from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CrossModalReparamLinear(nn.Linear):
    """
    Linear layer with cross-modal reparameterization.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        origin_layer (nn.Linear, optional): Original linear layer to initialize the weight and bias from. Default is None.
        aux_weight (torch.Tensor, optional): Auxiliary weight tensor. Default is None.
        is_aux_trainable (bool, optional): If set to False, the auxiliary weight will not be trainable. Default is True.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        origin_layer=None,
        aux_weight=None,
        is_aux_trainable=True,
    ):
        super().__init__(in_features, out_features, bias)
        self.cross_modal_scale = nn.Parameter(torch.zeros(1))
        assert (
            self.weight.size() == aux_weight.size()
        ), "Target weight and aux weight must have the same shape"
        self.aux_weight = aux_weight
        self.aux_weight.requires_grad_(is_aux_trainable)
        if origin_layer is not None:
            with torch.no_grad():
                self.weight.copy_(origin_layer.weight)
                self.bias.copy_(origin_layer.bias)

    def forward(self, input):
        """
        Forward pass of the CrossModalReparamLinear layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        weight = self.weight + self.cross_modal_scale * self.aux_weight
        return F.linear(input, weight, self.bias)


def cross_modal_ffn(
    ffn_original_linear: nn.Linear,
    ffn_auxiliar_linear: nn.Linear,
    dim: int,
    ff_mult: int,
    dropout: float,
    ffn_original_last_linear: nn.Linear,
    ffn_aux_last_linear: nn.Linear,
    *args,
    **kwargs,
):
    """
    Cross-modal feed-forward network.

    Args:
        ffn_original_linear (nn.Linear): Linear layer for the original modality.
        ffn_auxiliar_linear (nn.Linear): Linear layer for the auxiliary modality.
        dim (int): Dimension of the input.
        ff_mult (int): Multiplier for the hidden dimension.
        dropout (int): Dropout rate.
        ffn_original_last_linear (nn.Linear): Linear layer for the original modality in the last step.
        ffn_aux_last_linear (nn.Linear): Linear layer for the auxiliary modality in the last step.
        *args: Variable length arguments.
        **kwargs: Keyword arguments.

    Returns:
        nn.Sequential: Sequential model representing the cross-modal feed-forward network.
    """

    ffn_1st_rep_linear = CrossModalReParametrization(
        ffn_original_linear(dim, dim * ff_mult),
        ffn_auxiliar_linear(dim, dim * ff_mult),
    )

    ffn_2nd_linear = CrossModalReParametrization(
        ffn_original_last_linear(dim * ff_mult, dim),
        ffn_aux_last_linear(dim * ff_mult, dim),
    )

    return nn.Sequential(
        ffn_1st_rep_linear,
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(dim**ff_mult),
        nn.GELU(),
        ffn_2nd_linear,
        nn.LayerNorm(dim),
    )


def build_cross_modal_reparam_linear(origin_layer, aux_layer):
    assert origin_layer.weight.size() == aux_layer.weight.size()
    return CrossModalReparamLinear(
        in_features=origin_layer.in_features,
        out_features=origin_layer.out_features,
        origin_layer=origin_layer,
        bias=origin_layer.bias is not None,
        aux_weight=aux_layer.weight,
    )


def _get_attr_by_name(obj, attr_name):
    attrs = attr_name.split(".")
    for a in attrs:
        obj = obj.__getattr__(a)
    return obj


def _set_attr_by_name(obj, attr_name, attr_value):
    owner = obj
    attr_names = attr_name.split(".")
    if len(attr_names) > 1:
        for a in attr_names[:-1]:
            owner = owner.__getattr__(a)
    owner.__setattr__(attr_names[-1], attr_value)


def change_original_linear_to_reparam(target_module, aux_module, layer_name):
    origin_linear_layer = _get_attr_by_name(target_module, layer_name)
    aux_linear_layer = _get_attr_by_name(aux_module, layer_name)
    reparam_layer = build_cross_modal_reparam_linear(
        origin_linear_layer, aux_linear_layer
    )
    _set_attr_by_name(target_module, layer_name, reparam_layer)


def reparameterize_aux_into_target_model(
    target_model,
    aux_model,
    layer_names=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
    main_body_name="blocks",
):
    """
    Reparameterizes the auxiliary model into the target model by replacing specific layers with corresponding layers from the auxiliary model.

    Args:
        target_model (object): The target model to reparameterize.
        aux_model (object): The auxiliary model containing the replacement layers.
        layer_names (tuple, optional): The names of the layers to be replaced. Defaults to ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2").
        main_body_name (str, optional): The name of the main body of the models. Defaults to "blocks".
    """
    target_transformer_blocks = _get_attr_by_name(target_model, main_body_name)
    aux_transformer_blocks = _get_attr_by_name(aux_model, main_body_name)
    for target_block, aux_block in zip(
        target_transformer_blocks, aux_transformer_blocks
    ):
        for layer_name in layer_names:
            change_original_linear_to_reparam(
                target_block, aux_block, layer_name
            )


class CrossModalReParametrization(nn.Module):
    """
    A module for cross-modal reparametrization.

    Args:
        original_linear (nn.Linear): The original linear layer.
        auxiliary_linear (nn.Linear): The auxiliary linear layer.

    Attributes:
        cross_modal_scale (nn.Parameter): The scale parameter for cross-modal reparametrization.

    Methods:
        forward(x: Tensor) -> Tensor: Performs forward pass through the module.
        merge(): Merges the weights and biases of the original and auxiliary linear layers.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        auxiliary_linear: nn.Linear,
        linears: List[nn.Linear] = None,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.auxiliary_linear = auxiliary_linear
        self.cross_modal_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        combined_weight = (
            self.original_linear.weight
            + self.cross_modal_scale * self.auxiliary_linear.weight
        )
        return nn.functional.linear(
            x, combined_weight, self.original_linear.bias
        )

    def merge(self):
        self.original_linear.weight.data.add_(
            self.cross_modal_scale.item() * self.auxiliary_linear.weight.data
        )
        if (
            self.original_linear.bias is not None
            and self.auxiliary_linear.bias is not None
        ):
            self.original_linear.bias.data.add_(
                self.cross_modal_scale.item() * self.auxiliary_linear.bias.data
            )
