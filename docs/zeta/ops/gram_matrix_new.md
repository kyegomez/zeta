# gram_matrix_new

This feature is pivotal for capturing the correlation of features in the context of neural style transfer and texture synthesis. Understanding and utilizing the `gram_matrix_new` function enables users to implement and comprehend advanced neural network models that depend on feature correlations.


A Gram matrix represents the inner product of vectors which, in deep learning, typically correspond to flattened feature maps of a convolutional layer. Calculating Gram matrices is fundamental in style transfer algorithms, as the Gram matrix encapsulates texture information. By comparing Gram matrices of different images, networks can be trained to minimize the style differences between them, effectively transferring the style from one image to the other.

## `gram_matrix_new` Function Definition

Here is the formal definition and parameters of the `gram_matrix_new` function:

```python
def gram_matrix_new(y):
    """
    Computes the Gram matrix of a given tensor, often used in neural network algorithms to capture the correlation between features.

    The Gram matrix is calculated by performing an element-wise product between the feature maps followed by a summation over spatial dimensions.

    Parameters:
    - y (Tensor): A 4D tensor with shape (batch_size, channels, height, width) that represents the feature maps.

    Returns:
    - Tensor: A 3D tensor with shape (batch_size, channels, channels) representing the Gram matrix of the input tensor.
    """

    b, ch, h, w = y.shape
    return torch.einsum(
        "bchw,bdhw->bcd",
        [y, y]
    ) / (h * w)
```

## Explanation of the Functionality and Usage

The `gram_matrix_new` function takes a 4D tensor as input, which is the standard shape for batched image data in PyTorch, with dimensions for batch size, channels, height, and width. It uses the `einsum` function from the PyTorch library to compute the element-wise product and sum over spatial dimensions to calculate the Gram matrix. The function returns a 3D tensor where the batch dimension is retained, and the spatial correlation of the features is captured in a channels-by-channels matrix for each image in the batch.

## Detailed Usage Examples

Let's delve into three example usages of the `gram_matrix_new` function to understand it better in practical scenarios.

### Example 1: Basic Usage

```python
import torch
from zeta.ops import gram_matrix_new

# Simulated feature maps from a convolutional layer
feature_maps = torch.randn(1, 3, 64, 64)  # Simulating a single image with 3 channels

# Calculate the Gram matrix
gram_matrix = gram_matrix_new(feature_maps)

print(gram_matrix.shape)  # Output expected: (1, 3, 3)
```

In this basic usage example, we generate random feature maps to simulate the output of a convolutional layer for a single image with three channels. We then apply the `gram_matrix_new` function to calculate the Gram matrix.

### Example 2: Style Transfer Preparation

```python
import torch
import torchvision.models as models
from torchvision.transforms import functional as F
from PIL import Image
from zeta.ops import gram_matrix_new

# Load a pre-trained VGG model
vgg = models.vgg19(pretrained=True).features.eval()

# Load content and style images and preprocess them
content_img = Image.open('path/to/content/image.jpg')
style_img = Image.open('path/to/style/image.jpg')

# Preprocess images to match VGG input requirements
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
content_tensor = transform(content_img).unsqueeze(0)
style_tensor = transform(style_img).unsqueeze(0)

# Extract features from a specific layer in VGG
def get_features(image, model, layers=('conv_4',)):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

content_features = get_features(content_tensor, vgg)
style_features = get_features(style_tensor, vgg)

# Compute Gram matrix for style features
style_gram_matrix = {layer: gram_matrix_new(features) for (layer, features) in style_features.items()}

print(style_gram_matrix['conv_4'].shape)  # Output expected: (1, C, C)
```

In this example, we preprocess content and style images, extract their features using a VGG model, and then use the `gram_matrix_new` function to calculate the Gram matrix for the style image's features. This is a crucial step in a style transfer algorithm.

### Example 3: Optimizing a Neural Network for Style

```python
import torch
import torch.optim as optim
from zeta.ops import gram_matrix_new
from torchvision.models import vgg19

# Assume content_tensor, style_tensor, and their Gram matrices are already prepared as above

# Define a transformation network and initialize with random weights
transformation_net = YourTransformationNet()  # YourTransformationNet should be a PyTorch model that you have defined

# Define a loss function and optimizer
optimizer = optim.Adam(transformation_net.parameters(), lr=0.001)
mse_loss = torch.nn.MSELoss()

# Optimization loop
for epoch in range(num_epochs):
    # Generate transformed image from the content image
    transformed_img = transformation_net(content_tensor)
    
    # Extract features of the transformed image in the same way as for content and style images
    transformed_features = get_features(transformed_img, vgg)
    transformed_gram_matrix = gram_matrix_new(transformed_features['conv_4'])

    # Compute loss based on difference in Gram matrices
    style_loss = mse_loss(transformed_gram_matrix, style_gram_matrix['conv_4'])

    # Backpropagation and optimization
    optimizer.zero_grad()
    style_loss.backward()
    optimizer.step()
```

The third example demonstrates incorporating the `gram_matrix_new` function into an optimization loop for training a neural network to perform style transfer. The network is optimized to minimize the difference between the Gram matrices of the transformed and style images.

## Arguments and Methods Summary in Markdown Table

| Argument       | Type     | Description                                       | Default Value | Required |
| -------------- | -------- | ------------------------------------------------- | ------------- | -------- |
| `y`            | Tensor   | A 4D input tensor with shape (b, ch, h, w).       | None          | Yes      |

| Method              | Returns  | Description                                      |
| ------------------- | -------- | ------------------------------------------------ |
| `gram_matrix_new`   | Tensor   | Computes a 3D gram matrix from the input tensor. |

## Additional Information and Tips

- When calculating the Gram matrix of large feature maps, be aware that this operation can be memory-intensive, as the computation requires a quadratic amount of memory relative to the number of channels.
- To improve computational efficiency, consider converting input tensors to half-precision (`torch.float16`) if your hardware support.

## References and Resources

1. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
2. Neural Style Transfer: A Review: https://arxiv.org/abs/1705.04058
3. Visualizing and Understanding Convolutional Networks: https://arxiv.org/abs/1311.2901
