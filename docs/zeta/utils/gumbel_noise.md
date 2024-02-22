# gumbel_noise

# gumbel_noise Function Documentation

## Function Definition

`gumbel_noise(t)`

The `gumbel_noise` function generates Gumbel-distributed noise given a tensor object `t`. The Gumbel distribution, often used in modeling extremes, is used here to generate noise with similar characteristics. To add randomness or noise to your models, this function is crucial especially when working with GANs, Variational Autoencoders or other stochastic architectures where random sampling is a key component.


## Parameters:

| Parameter     | Type                                                 | Description                                                  |
|---------------|------------------------------------------------------|--------------------------------------------------------------|
| `t`           | A tensor object                                      | Any PyTorch's tensor onto which noise would be generated     |

## Returns:

`noise`: A tensor object of the same shape as `t`, comprising of noise data sampled from Gumbel distribution.

## Function Usage

Before we jump onto the function usage, here's a brief about the Gumbel Distribution: The Gumbel Distribution, also known as Smallest Extreme Value (SEV) or Type I Extreme Value distribution, is a continuous probability distribution named after Emil Julius Gumbel. It is widely used in modeling extreme value problems in fields such as hydrology, structural engineering and climate data analysis.

Now let's go through a few examples illustrating the usage of `gumbel_noise` function:

### Import Necessary Libraries

```python
import torch
```

#### Example 1: Generation of Gumbel-Distributed Noise for a 1D Tensor Object

```python
# Define a tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Generate Gumbel noise
gumbel_noise_data = gumbel_noise(tensor)

# Output
print(gumbel_noise_data)
```

In this example, gumbel_noise_data is a tensor of the same size as the input tensor, but filled with noise sampled from the Gumbel distribution.

#### Example 2: Generation of Gumbel-Distributed Noise for a 2D Tensor Object

```python
# Define a 2D tensor
tensor_2D = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Generate Gumbel noise
gumbel_noise_data2D = gumbel_noise(tensor_2D)

# Output
print(gumbel_noise_data2D)
```

In this example, gumbel_noise_data2D is a 2D tensor of the same size as the input tensor, but filled with noise sampled from the Gumbel distribution.

#### Example 3: Generation of Gumbel-Distributed Noise for a 3D Tensor Object

```python
# Define a 3D tensor
tensor_3D = torch.rand((2, 2, 2))

# Generate Gumbel noise
gumbel_noise_data3D = gumbel_noise(tensor_3D)

# Output
print(gumbel_noise_data3D)
```

In this example, gumbel_noise_data3D is a 3D tensor of the same size as the input tensor, but filled with noise sampled from the Gumbel distribution.

This function, `gumbel_noise`, can be utilized in modelling various Machine Learning tasks - such as classification and generation tasks, and in building deep learning architectures, where learning from noise is beneficial, such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs) etc.

## Notes and Additional Information

When dealing with statistical modelling problems in Machine Learning, it's quite important and frequent to add statistical noise into the data. Because random noise makes the model more robust and generalizable. There are many types of noise that can be added into the data, Gumbel noise being one of them.

The purpose of adding this Gumbel noise is to provide a stochastic element to the PyTorch tensor, resulting in a distribution of values which can be manipulated or studied. The Gumbel noise added onto `t` by `gumbel_noise` essentially provides a simple way of getting a version of `t` that has been noise-adjusted. This can be important for methods which need a stochastic element or for testing the robustness of different architectures to noise.

It's worth noting that the Gumbel distribution has heavier tails than the normal distribution, so adding Gumbel noise to a variable will add extreme values (i.e., very large or very small numbers) more frequently than adding Gaussian noise. This means that using Gumbel noise can be a good way to test the stability and robustness of your model: if your model works well when you add Gumbel noise to the inputs, it's likely to also perform
