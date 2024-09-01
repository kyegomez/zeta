# SLERPModelMerger

- **Description**: 
SLERPModelMerger is a Python class that performs model merging using Spherical Linear Interpolation (SLERP). Interpolation is a process of finding a value between two points on a line or curve to create new geometries. Spherical Linear Interpolation (SLERP) is a method of interpolation where the model weights are visualized on a hypersphere, and the interpolated weight is obtained by moving along the geodesic (or the shortest path) on the hypersphere. This class is implemented under the PyTorch framework.

The class can blend or interpolate the weights of two trained models, allowing one to create an ensemble or composite model of the input models, essentially capturing the strengths of both. In ML terminology, this can be thought of as a "committee machine" where transformations applied to input data by multiple models are combined to produce a single output. This method is known to improve the robustness and performance of models, especially in scenarios where the strength of individual models varies across different sections of the input space.

- **Class Definition**: 

Here is the class definition:

```python
class SLERPModelMerger(nn.Module):
    @enforce_types
    def __init__(self, model1: nn.Module, model2: nn.Module, t: float = 0.5):

    def merge(self) -> nn.Module:

    @staticmethod
    @enforce_types
    def _slerp(w1: Tensor, w2: Tensor, t: float) -> Tensor:

    @staticmethod
    @enforce_types
    def _copy_model_structure(model: nn.Module) -> nn.Module:
```

- **Parameters:**
    `model1` and `model2` are instances of PyTorch's neural network models (such as instances of `nn.Linear, nn.Conv2d` etc.) between which weights' interpolation is to be done. The parameter `t` is the interpolation parameter that ranges from 0 (model1) to 1 (model2), indicating the weightage given to the two models during interpolation. Hence, for t=0, the resulting model would be the same as model1, and for t=1, the resulting model would be the same as model2.

- **Methods:**

    - `merge()` : This method merges the input models (`model1` and `model2`), according to the interpolation parameter `t`. The merging is done by interpolating the weights of the two models using Spherical Linear Interpolation (SLERP).
    
    - `_slerp(w1: Tensor, w2: Tensor, t: float) -> Tensor:` : This method performs Spherical Linear Interpolation (SLERP) between two tensors.
    
    - `_copy_model_structure(model: nn.Module) -> nn.Module:` : This method creates a new instance of a model with the same structure as the given model.

- **Usage:**

The following code shows how to use the SLERPModelMerger class to merge two PyTorch models (in this case two linear models):

```python
import torch.nn as nn

from zeta.nn import SLERPModelMerger

model1 = nn.Linear(10, 10)
model2 = nn.Linear(10, 10)

merger = SLERPModelMerger(model1, model2, 0.5)
merged_model = merger.merge()

# This will output the merged state_dict
print(merged_model.state_dict())
```

The prints statement will output the state_dict of the merged model. The state_dict is a Python dictionary that maps each layer to its corresponding parameters (tensors). 

The weightage given to the two models for interpolation is specified by the interpolation parameter `t`. As t ranges from 0 to 1, we can see the merged model evolve from model1 to model2. Thus, by changing `t` we can generate a spectrum of models from model1 to model2.

This gives us a strategy to generate an ensemble of models by interpolating between two carefully chosen base models. This ensemble could then be used for model selection or for creating a more robust composite model.

- **References:**
    
    - Ken Shoemake. Animating rotation with quaternion curves. In ACM SIGGRAPH Computer Graphics, volume 19, pp. 245–254. ACM, 1985.

Remarks: Remember, while PyTorch models accept parameters as single arguments to their constructors, this is not the case with all models. Some models might accept parameters as lists, sets, or other non-single-parameter-type objects. As such, additional pre-processing or configuration might be needed if using those models with SLERPModelMerger. Try these different configurations and methods to find the one that best suits your requirements.
