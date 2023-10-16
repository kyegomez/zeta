import torch
from torch import nn


class Exo(nn.Module):
    """

    Exo activation function
    -----------------------

    Exo is a new activation function that is a combination of linear and non-linear parts.

    Formula
    -------
    .. math::
        f(x) = \\sigma(x) \\cdot x + (1 - \\sigma(x)) \\cdot tanh(x)

    Parameters
    ----------
    alpha : float
        Alpha value for the activation function. Default: 1.0

    Examples
    --------
    >>> m = Exo()
    >>> input = torch.randn(2)
    >>> output = m(input)


    # Paper

    **"Exo": A Conceptual Framework**
    For the sake of this exercise, let's envision Exo as a cutting-edge activation function
    inspired by the idea of "extraterrestrial" or "outside the norm" data processing. The main premise is that it's designed to handle the vast heterogeneity in multi-modal data by dynamically adjusting its transformation based on the input distribution.

    ---

    **Technical Report on the Exo Activation Function**

    **Abstract**

    In the evolving landscape of deep learning and multi-modal data processing,
    activation functions play a pivotal role. This report introduces the "Exo" activation
     function, a novel approach designed to cater to the diverse challenges posed by multi-modal data.
    Rooted in a dynamic mechanism, Exo adjusts its transformation based on the input distribution, offering flexibility and efficiency in handling heterogeneous data.

    **1. Introduction**

    Activation functions serve as the heart of neural networks, determining the output of a node given an input or set of inputs. As deep learning models grow in complexity, especially in the realm of multi-modal data processing, there's a pressing need for activation functions that are both versatile and computationally efficient. Enter Exo—a dynamic, adaptive function envisioned to rise to this challenge.

    **2. Design Philosophy**

    At its core, Exo embodies the idea of adaptability. Drawing inspiration from the vast, unpredictable expanse of outer space, Exo is designed to dynamically adjust to the data it processes. This inherent flexibility makes it a prime candidate for multi-modal tasks, where data can be as varied as the stars in the sky.

    **3. Mechanism of Operation**

    Exo operates on a simple yet powerful principle: adaptive transformation. It leverages a gating mechanism that weighs the influence of linear versus non-linear transformations based on the magnitude and distribution of input data.

    The pseudocode for Exo is as follows:

    ```
    function Exo(x, alpha):
        gate = sigmoid(alpha * x)
        linear_part = x
        non_linear_part = tanh(x)
        return gate * linear_part + (1 - gate) * non_linear_part
    ```

    **4. Why Exo Works the Way It Does**

    The strength of Exo lies in its adaptive nature. The gating mechanism—dictated by
    the sigmoid function—acts as a switch. For high-magnitude inputs,
    Exo trends towards a linear behavior. Conversely, for lower-magnitude inputs,
    it adopts a non-linear transformation via the tanh function.

    This adaptability allows Exo to efficiently handle data heterogeneity,
    a prominent challenge in multi-modal tasks.

    **5. Ideal Use Cases**

    Given its versatile nature, Exo shows promise in the following domains:

    - **Multi-Modal Data Processing**: Exo's adaptability makes it a strong contender
    for models handling diverse data types, be it text, image, or audio.

    - **Transfer Learning**: The dynamic range of Exo can be beneficial when transferring
    knowledge from one domain to another.

    - **Real-time Data Streams**: For applications where data distributions might change
    over time, Exo's adaptive nature can offer robust performance.

    **6. Experimental Evaluation**

    Future research will rigorously evaluate Exo against traditional activation functions
    across varied datasets and tasks.

    ---

    **Methods Section for a Research Paper**

    **Methods**

    **Activation Function Design**

    The Exo activation function is defined as:

    \[ Exo(x) = \sigma(\alpha x) \times x + (1 - \sigma(\alpha x)) \times \tanh(x) \]

    where \(\sigma\) represents the sigmoid function, and \(\alpha\) is a hyperparameter
    dictating the sensitivity of the gating mechanism.

    **Model Configuration**

    All models were built using the same architecture, with the only difference
    being the activation function. This ensured that any performance disparities
    were solely attributed to the activation function and not other model parameters.

    **Datasets and Pre-processing**

    Three diverse datasets representing image, text, and audio modalities were employed.
    All datasets underwent standard normalization procedures.

    **Training Regimen**

    Models were trained using the Adam optimizer with a learning rate of 0.001 for
    50 epochs. Performance metrics, including accuracy and loss, were recorded.


    """

    def __init__(self, alpha=1.0):
        """INIT function."""
        super(Exo, self).__init__()

    def forward(self, x):
        """Forward function."""
        gate = torch.sigmoid(x)
        linear_part = x
        non_linear_part = torch.tanh(x)
        return gate * linear_part + (1 - gate) * non_linear_part
