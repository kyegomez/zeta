import torch
import torch.nn as nn


class LogGammaActivation(torch.autograd.Function):
    """
    PulSar Activation function that utilizes factorial calculus

    PulSar Activation function is defined as:
        f(x) = log(gamma(x + 1))
    where gamma is the gamma function

    The gradient of the PulSar Activation function is defined as:
        f'(x) = polygamma(0, x + 2)
    where polygamma is the polygamma function

    Methods:
        forward(ctx, input): Computes the forward pass
        backward(ctx, grad_output): Computes the backward pass
    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the PulSar Activation function

        """
        # compute forward pass
        gamma_value = torch.lgamma(input + 1)
        ctx.save_for_backward(input, gamma_value)
        return gamma_value

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the PulSar Activation function
        """
        # compute gradient for backward pass
        input, gamma_value = ctx.saved_tensors
        polygamma_val = torch.polygamma(0, input + 2)
        return polygamma_val * grad_output


class Pulsar(nn.Module):
    """
        Pulsar Activation function that utilizes factorial calculus

        Pulsar Activation function is defined as:
            f(x) = log(gamma(x + 1))
        where gamma is the gamma function


        Usage:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        pulsar = Pulsar()
        y = pulsar(x)
        print(y)
        y = y.backward(torch.ones_like(x))


        I apologize for the oversight. Let's dive into a technical report on a hypothetical "Pulsar" activation function. Given that "Pulsar" as an activation function doesn't exist (as of my last training cut-off in January 2022), this will be a fictional report, but I'll approach it in the style of a technical paper.

    ---

        **Technical Report on the Pulsar Activation Function**

        ---

        ### **Abstract:**

        In the realm of deep learning, activation functions play a pivotal role in introducing non-linearity to the model, enabling it to learn intricate patterns from data. This report introduces a new activation function named "Pulsar". The underlying mechanics, principles, and potential applications of Pulsar are discussed, offering insights into its efficiency in various machine learning tasks.

        ---

        ### **1. Introduction:**

        Activation functions form the backbone of neural networks, determining the output of a neuron given an input or set of inputs. Pulsar, inspired by its celestial namesake, is designed to oscillate and adapt, providing dynamic activation thresholds.

        ---

        ### **2. Background:**

        Popular activation functions such as ReLU, Sigmoid, and Tanh have set the standard in deep learning models. While they have their advantages, there are challenges like the vanishing and exploding gradient problems. Pulsar aims to address some of these challenges by offering a dynamic threshold and adaptive behavior.

        ---

        ### **3. How Pulsar Works:**

        #### **3.1 Mechanism of Activation:**

        Pulsar's main innovation lies in its oscillatory behavior. Instead of providing a static threshold or curve like ReLU or Sigmoid, Pulsar's activation threshold changes based on the input's context, oscillating like a pulsar star.

        #### **3.2 Mathematical Representation:**

        Given an input `x`, the Pulsar activation, `P(x)`, can be represented as:

        \[ P(x) = x \times \sin(\alpha x + \beta) \]

        Where:
        - \( \alpha \) and \( \beta \) are parameters that control the oscillation frequency and phase. They can be learned during training or set as hyperparameters.

        ---

        ### **4. Why Pulsar Works the Way It Does:**

        #### **4.1 Dynamic Thresholding:**

        The oscillatory nature allows Pulsar to dynamically threshold activations, making it adaptable. In contexts where other activations might suffer from saturation or dead neurons, Pulsar's oscillation ensures continuous activation adjustment.

        #### **4.2 Mitigating Common Problems:**

        The design aims to mitigate the vanishing and exploding gradient problems by ensuring that the gradient is neither too small nor too large across a broad range of input values.

        ---

        ### **5. Applications and Tasks for Pulsar:**

        #### **5.1 Complex Pattern Recognition:**

        Due to its dynamic nature, Pulsar can be particularly useful for datasets with intricate and overlapping patterns.

        #### **5.2 Time-Series and Signal Processing:**

        The oscillatory behavior might resonate well with time-series data, especially those with underlying periodic patterns.

        #### **5.3 Regular Feedforward and Convolutional Networks:**

        Pulsar can serve as a drop-in replacement for traditional activations, potentially offering better performance in some contexts.

        ---

        ### **6. Methods:**

        #### **6.1 Model Configuration:**

        Neural networks with three hidden layers were employed. Each layer contained 128 neurons. We utilized Pulsar as the activation function for one set and compared it with ReLU, Sigmoid, and Tanh activations.

        #### **6.2 Dataset:**

        Experiments were conducted on the MNIST dataset. The dataset was split into 60,000 training images and 10,000 testing images.

        #### **6.3 Training:**

        Models were trained using the Adam optimizer with a learning rate of 0.001. A batch size of 64 was used, and models were trained for 20 epochs.

        #### **6.4 Evaluation Metrics:**

        Models were evaluated based on accuracy, loss convergence speed, and gradient flow behavior.

        ---

        ### **7. Conclusion:**

        The Pulsar activation function, with its unique oscillatory design, offers a fresh perspective on neuron activations. Initial experiments show promise, especially in tasks with complex patterns. Further research and optimization can potentially cement its place alongside traditional activation functions.

        ---

        (Note: This is a fictional report. The Pulsar activation function, its properties, and the described results are all hypothetical and for illustrative purposes only.)



    """

    def forward(self, x):
        """
        Forward pass of the PulSar Activation function
        """
        return LogGammaActivation.apply(x)


class PulsarNew(nn.Module):
    def __init__(self, alpha=0.01, beta=0.5):
        super(PulsarNew, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor):
        # compute leaky rely
        leaky = self.alpha * x

        # compute saturated tanh component
        saturated = self.beta + (1 - self.beta) * torch.tanh(x - self.beta)

        # compute based on conditions
        return torch.where(
            x < 0, leaky, torch.where(x < self.beta, x, saturated)
        )


x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
pulsar = PulsarNew()
pulsar(x)
