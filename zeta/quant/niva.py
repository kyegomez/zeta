from typing import List, Type, Union

import torch
from torch import nn


def niva(
    model: nn.Module,
    model_path: str = None,
    output_path: str = None,
    quant_type: str = "dynamic",
    quantize_layers: Union[List[Type[nn.Module]], None] = None,
    dtype: torch.dtype = torch.qint8,
    *args,
    **kwargs,
):
    """Niva: Quantize a model.

    Args:
        model (nn.Module): _description_
        model_path (str, optional): _description_. Defaults to None.
        output_path (str, optional): _description_. Defaults to None.
        quant_type (str, optional): _description_. Defaults to "dynamic".
        quantize_layers (Union[List[Type[nn.Module]], None], optional): Quantize layers. Defaults to None.
        dtype (torch.dtype, optional): _description_. Defaults to torch.qint8.

    Raises:
        TypeError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
        ValueError: _description_
        ValueError: _description_

    Examples:
    >>> import torch
    >>> from zeta.quant import niva
    >>> from zeta.nn import QFTSPEmbedding
    >>> model = QFTSPEmbedding(100, 100)
    >>> niva(
    ...     model,
    ...     quant_type="static",
    ...     dtype=torch.qint8,
    ...     quantize_layers=[nn.Embedding],
    ...     model_path="model.pt",
    ...     output_path="model_quantized.pt"
    ... )

    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if model_path is None:
        raise ValueError("model_path must be specified")
    if output_path is None:
        raise ValueError("output_path must be specified")
    if quant_type not in ["static", "dynamic"]:
        raise ValueError("quant_type must be either static or dynamic")
    if quantize_layers is not None:
        if not isinstance(quantize_layers, list):
            raise TypeError("quantize_layers must be a list")
        for layer in quantize_layers:
            if not isinstance(layer, type):
                raise TypeError("quantize_layers must be a list of types")
            if not issubclass(layer, nn.Module):
                raise TypeError(
                    "quantize_layers must be a list of types that are"
                    " subclasses of torch.nn.Module"
                )
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if dtype not in [torch.qint8, torch.quint8]:
        raise ValueError("dtype must be either torch.qint8 or torch.quint8")

    # Load the model
    model.load_state_dict(torch.load(model_path))

    # Ensure model is in eval model
    model.eval()

    # Apply quantization
    if quant_type == "dynamic":
        if quantize_layers is None:
            raise ValueError(
                "quantize_layers must be specified for dynamic quantization"
            )
        model = torch.quantization.quantize_dynamic(
            model, quantize_layers, dtype=dtype, *args, **kwargs
        )
    elif quant_type == "static":
        model.qconfig = torch.quantization.get_default_qconfig(dtype=dtype)
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

    # Save the model
    torch.save(model.state_dict(), output_path)
