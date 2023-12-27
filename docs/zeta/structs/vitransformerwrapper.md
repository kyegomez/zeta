# ViTransformerWrapper

## Introduction

`ViTransformerWrapper` is a PyTorch module that is part of the Zeta library. It essentially serves as a wrapper encapsulating the entirety of a Vision Transformer (ViT) model's architecture and functionality. As the name suggests,  this model is a Transformer that processes images. It treats an image as a sequence of image patches, much like how a regular Transformer treats a sentence as a sequence of words or subwords.

Since it's structurally a Transformer, `ViTransformerWrapper` leverages the multi-head self-attention mechanism which allows it to process image patches globally instead of locally. This gives `ViTransformerWrapper` the capability to reason about global image features and their intricate interrelations, a task that CNNs aren't built for.

## Class Definition

The `ViTransformerWrapper` class inherits from PyTorch's `nn.Module` class which is the base class for all neural network modules. This class also has a layer called `attn_layers` which must be an `Encoder` object, this `Encoder` is a standard Transformer encoder.

```python
class ViTransformerWrapper(nn.Module):
    def __init__(self, *, image_size, patch_size, attn_layers, channels=3, num_classes=None, post_emb_norm=False, emb_dropout=0.0):
    def forward(self, img, return_embeddings=False):
```

### Parameters

| Parameter     | Type | Description |
|---------------|------|-------------|
| image_size    | int  | Size of the image. The dimension must be divisible by `patch_size`. |
| patch_size    | int  | Size of the image patches. |
| attn_layers   | Encoder  | Transformer encoder which will be used as the attention layers. |
| channels      | int (default is 3)  | Number of channels in the image. |
| num_classes   | int (optional)  | Number of classes in the classification task. If `None`, the model will output raw embeddings. |
| post_emb_norm | bool (default is `False`) | If `True`, enables normalization of embeddings after they are generated. |
| emb_dropout   | float (default is 0.0) | Dropout rate for the embeddings. |

### Attributes

| Attribute    | Type | Description |
|--------------|------|-------------|
| training | bool | Represents whether the module is in training mode or evaluation mode. |

Attributes, methods and submodules assigned in the `__init__` method are registered in the module and will have their parameters converted too when you call `to()`, etc.

### Method: `forward`

The `forward` method is called when we execute the `ViTransformerWrapper` instance as a function. It feeds an image through the model and computes the forward pass. If `return_embeddings` is set to `True`, the method will output raw embeddings, otherwise it will output the predictions of the model, using the `mlp_head` which is a fully-connected layer applied after the Transformer layers.

Parameters:

- `img` (Tensor): Input image.
- `return_embeddings` (bool, optional): If `True`, the method returns raw embeddings. If `False` (default), the method returns the class predictions.

## Usage Examples

Here are three usage examples:

### Example 1: Basic Usage

```python
from zeta.structs import ViTransformerWrapper, Encoder

# create a Transformer encoder instance
encoder = Encoder(dim=128, depth=12)

# define the wrapper with the encoder
wrapper = ViTransformerWrapper(image_size=224, patch_size=16, attn_layers=encoder)

# sample image
img = torch.randn(1, 3, 224, 224)

# output of the model
out = wrapper(img)
```

In this example, we first create an instance of a Transformer encoder with a dimension of 128 and a depth of 12. Then we instanstiate the `ViTransformerWrapper` with an image size of 224, a patch size of 16 and the previously created Transformer encoder. Afterwards, we simulate an image input of torch size (1, 3, 224, 224) and feed it through the model by calling `wrapper(img)`, the resulting `out` is the output of the model.

### Example 2: Training Loop

```python
from zeta.structs import ViTransformerWrapper, Encoder

# create a Transformer encoder instance
encoder = Encoder(dim=128, depth=12)

# define the wrapper with the encoder and the number of classes
model = ViTransformerWrapper(image_size=224, patch_size=16, attn_layers=encoder, num_classes=10)

# define a loss function
criterion = nn.CrossEntropyLoss()

# define an optimizer
optimizer = torch.optim.Adam(model.parameters())

# sample inputs and targets
inputs = torch.randn(32, 3, 224, 224)
targets = torch.randint(0, 10, [32])

# training loop
for i in range(100):

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)

    # compute the loss
    loss = criterion(outputs, targets)

    # backward pass and optimize
    loss.backward()
    optimizer.step()

    # print statistics
    print('loss: {:.4f}'.format(loss.item()))
```

This example shows a basic training loop for the `ViTransformerWrapper`. In this training loop, we use a cross entropy loss and Adam as the optimizer. The loop goes for 100 iterations, in each iteration it firstly zeroes the gradients, conducts forward pass to compute the model's output, then computes the loss based on the output and the ground truth, backpropagates the gradients and finally updates the model's parameters according to the Adam optimizer. The loss is printed out at every iteration.

### Example 3: Embeddings

```python
from zeta.structs import ViTransformerWrapper, Encoder

# create a Transformer encoder instance
encoder = Encoder(dim=128, depth=12)

# define the wrapper with the encoder
model = ViTransformerWrapper(image_size=224, patch_size=16, attn_layers=encoder)

# sample inputs
inputs = torch.randn(1, 3, 224, 224)

# compute the embeddings
embeddings = model(inputs, return_embeddings=True)
```

In this example, the `ViTransformerWrapper` returns raw embeddings since `return_embeddings` is set to `True`. The returned `embeddings` can then be used for other tasks such as clustering or nearest neighbours search.

## Additional Information

The `ViTransformerWrapper` class assumes that you're working with square images, i.e. height equals width. Be sure to resize your images appropriately or pad them if they are not originally square.

Also, the `mlp_head` output layer is initialized as an `nn.Identity` layer if `num_classes` is not specified, meaning the Transformer's output embeddings will be passed through without transformation.

Furthermore, the model relies on 2D convolutions, layer normalization and linear transformations, making it applicable to a wide range of tasks involving image data beyond image classification, such as object detection and instance segmentation, given suitable adjustments. 

Lastly, vision transformers are computationally expensive and use significantly more memory than their CNN counterparts since self-attention operates in quadratic space and time. Consider this if using a vision transformer in your project.

## External Resources

- For further understanding on Transformers, you can read the following paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- For the original Vision Transformer paper, you can read: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- To know more about the implementation of the transformer model, consider reading the [Transformers Module in PyTorch](https://pytorch.org/docs/stable/nn.html#transformer-layers) documentation.
- For more tutorials and examples using PyTorch, you can check out their [tutorials page](https://pytorch.org/tutorials/).
