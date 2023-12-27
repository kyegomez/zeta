# top_k

# Module/Function Name: top_k

```python
def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs
```

The `top_k` function is utility function that is used to retrieve the top k logits based on a threshold. It takes in the logits and a threshold value, picks out the top k logits that meet the threshold, and then returns those logits.

## Parameters
| Parameter | Type | Description | Default |
| :---      | :--- | :---        | :---    |
| logits   | Tensor | A rank 1 tensor representing the logits you want to filter | Required |
| thres | float | A float representing the threshold for filtering, the default value is 0.9 | 0.9 |

## Returns
| Return | Type | Description |
| :---   | :--- | :---    |
| probs | Tensor | The tensor after being filtered |

## Usage Examples

Now, let's go through a few examples of how you can use the `top_k` function.

### Example 1: Basic usage

In the most basic usage, you would pass a tensor of logits and receive a filtered tensor.

```python
import torch
from math import ceil
def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

logits = torch.tensor([0.1, 0.4, 0.3, 0.2, 0.5])
probs = top_k(logits)
print(probs) 
```

### Example 2: Changing the Threshold

The threshold value can be adjusted according to your requirements. A higher threshold may result in values being included that would otherwise be excluded.

```python
import torch
from math import ceil
def top_k(logits, thres=0.8):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

logits = torch.tensor([0.1, 0.4, 0.3, 0.2, 0.5])
probs = top_k(logits)
print(probs) 
```

### Example 3: Using a Different Tensor

The input tensor can be changed as needed. The only requirement is that the tensor should be a 1D tensor.

```python
import torch
from math import ceil
def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs

logits = torch.tensor([0.1, 0.4, 0.7, 0.2, 0.5])
probs = top_k(logits)
print(probs) 
```

## Additional Information and Tips:

- The function `top_k` makes use of the `torch.topk()` function to find the top k values in the tensor and returns these values and their respective indices.
- The indices are used with the `torch.Tensor.scatter_()` function to replace the selected elements in a new tensor filled with `-inf` along the specified dimension with the specified value.
  
## References:

- For more information about the functions used, refer to the PyTorch documentation:
  - [torch.topk()](https://pytorch.org/docs/stable/generated/torch.topk.html)
  - [torch.Tensor.scatter_()](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)
