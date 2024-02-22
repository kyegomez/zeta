# Zeta.nn.modules.AverageModelMerger Documentation

## Introduction

The AverageModelMerger class, found in the zeta.nn.modules library, is a simple yet powerful class to merge multiple models by averaging their weights. It offers a straightforward way to combine models trained in different stages, such as instruction and alignment tuning, leading to improved model performance in certain circumstances.

## Class Definition: AverageModelMerger

```python
class AverageModelMerger:
    """
    A class to merge multiple models by averaging their weights.

    Attributes:
    models (List[nn.Module]): A list of PyTorch models to be merged.

    Examples::- Example usage:
    model1 = nn.Linear(in_features=10, out_features=10)
    model2 = nn.Linear(in_features=10, out_features=10)
    model3 = nn.Linear(in_features=10, out_features=10)
    merge = AverageModelMerger([model1, model2, model3])
    merged_model = merge.merge_models()
    print(merged_model)
    """
```

### Class Parameters:

| Parameters | Data Type     | Default Value | Description |
|------------|---------------|---------------|-------------|
| models     | List[nn.Module]     | N/A           | List of PyTorch models to be merged

### Class Methods:

| Method Name       | Description | Parameters | Returns |
|-------------------|-------------|------------|---------|
| `__init__(self, models: List[nn.Module])`| Initializes the AverageModelMerger with a list of models. | models (List[nn.Module]) | None |
| `merge_models(self)` | Merges the models by averaging their weights. | None | A new model with averaged weights. | 
| `_copy_model_structure(model: nn.Module)` | Creates a new instance of a model with the same structure as the given model. | model (nn.Module) | A new model with the same structure. | 

### Constructor `__init__(self, models: List[nn.Module])`

Initializes an instance of the AverageModelMerge class. It takes a list of PyTorch models as input which are to be merged later using the `merge_models` method. 

- **models (List[nn.Module])**: Models to be merged.

### Method `merge_models(self) -> nn.Module`

This function merges the models by averaging the weights of the PyTorch models. 

**Returns**

nn.Module: A new model with averaged weights.

### Method `_copy_model_structure(self, model: nn.Module) -> nn.Module`

This function creates a new instance of a model with exactly the same structure as the given model.

**Parameters**
- **model (nn.Module)**: The model whose structure is to be copied.

**Returns**

nn.Module: A new model with exactly the same structure.

## Examples of Usage:

### Example 1
```python
import torch.nn as nn

from zeta.nn.modules import AverageModelMerger

# Define models
model1 = nn.Linear(in_features=10, out_features=10)
model2 = nn.Linear(in_features=10, out_features=10)
model3 = nn.Linear(in_features=10, out_features=10)

# Initialize AverageModelMerger
merger = AverageModelMerger([model1, model2, model3])

# Merge models
merged_model = merger.merge_models()

# Print merged model
print(merged_model)
```

### Example 2
```python
import torch.nn as nn

from zeta.nn.modules import AverageModelMerger

# Define models
model1 = nn.Conv2d(3, 6, 5)
model2 = nn.Conv2d(3, 6, 5)
model3 = nn.Conv2d(3, 6, 5)

# Initialize AverageModelMerger
merger = AverageModelMerger([model1, model2, model3])

# Merge models
merged_model = merger.merge_models()

# Print merged model
print(merged_model)
```

### Example 3
```python
import torch.nn as nn

from zeta.nn.modules import AverageModelMerger

# Define models
model1 = nn.CrossEntropyLoss()
model2 = nn.CrossEntropyLoss()
model3 = nn.CrossEntropyLoss()

# Initialize AverageModelMerger
merger = AverageModelMerger([model1, model2, model3])

# Merge models
merged_model = merger.merge_models()

# Print merged model
print(merged_model)
```

All the examples above demonstrate the basic usage of this class. In cases where you have multiple trained models (e.g., resultant from a k-fold cross-validation or models trained on different datasets), you can use this class to merge or average their weights. The resultant model will carry averaged weights, giving a balanced representation of all the models.
