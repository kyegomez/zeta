import triton
import triton.language as tl


class Functions:
    @staticmethod
    @triton.jit
    def tanh_activation_kernel(
        x_ptr,
        out_ptr,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Applies the hyperbolic tangent (tanh) activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        exp2x = tl.exp(2 * x)
        output = 1 - 2 / (exp2x + 1)
        tl.store(out_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def hard_tanh_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the hard tanh activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        shape_condition = tl.where(x < -1, -1, x)
        output = tl.where(x > 1, 1, shape_condition)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def relu_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the rectified linear unit (ReLU) activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.maximum(0, x)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def relu6_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the rectified linear unit 6 (ReLU 6) activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.minimum(tl.maximum(x, 0), 6.0)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def leaky_relu_activation_kernel(
        x_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the LeakyReLU activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.maximum(x, alpha * x)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def smooth_relu_activation_kernel(
        x_ptr, output_ptr, n_elements, beta, BLOCK_SIZE: tl.constexpr
    ):
        """
        Convolution of ReLU with a box, transition region widens, the loss surface becomes smoother
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.where(x >= beta, x, 0.0)
        output = tl.where(
            tl.abs(x) <= beta, ((x + beta) * (x + beta) / (4.0 * beta), output)
        )

        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def softsign_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the softsign activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = x / (tl.abs(x) + 1)
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def softplus_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the softplus activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = tl.log(1 + tl.exp(x))
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def sigmoid_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the sigmoid activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = 1 / (1 + tl.exp(-x))
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def hard_sigmoid_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the hard sigmoid activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        x_plus_3 = x + 3.0
        relu6_result = tl.minimum(tl.maximum(x_plus_3, 0), 6.0)
        output = relu6_result / 6.0
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def silu_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the Sigmoid-weighted Linear Unit (SiLU) activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        output = x * (1 / (1 + tl.exp(-x)))
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def hard_silu_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the hard SiLU activation function to element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        x_plus_3 = x + 3.0
        relu6_result = tl.minimum(tl.maximum(x_plus_3, 0), 6.0)
        hard_sigmoid_output = relu6_result / 6.0
        output = x * hard_sigmoid_output
        tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def softmax_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the softmax activation function to the input tensor along the specified axis
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        max_x = tl.maximum(x, 0)
        x -= max_x
        exp_x = tl.exp(x)
        sum_exp_x = tl.sum(exp_x)
        output = exp_x / sum_exp_x
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def gelu_activation_kernel(
        x_ptr, output_ptr, approximation, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the input tensor
        """
        idx = tl.program_id(0)
        block_st = idx * BLOCK_SIZE
        offsets = block_st + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        if approximation is True:
            output = (
                0.5
                * x
                * (
                    1
                    + tl.libdevice.tanh(
                        tl.libdevice.sqrt(2.0 / 3.141592653589793)
                        * (x + 0.044715 * x * x * x)
                    )
                )
            )
            tl.store(output_ptr + offsets, output, mask=mask)
        else:
            output = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
            tl.store(output_ptr + offsets, output, mask=mask)

    @staticmethod
    @triton.jit
    def swiglu_activation_kernel(
        x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
        """
        Applies the SwiGLU activation function to the input tensor
        """
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements // 2
        f = tl.load(x_ptr + idx * 2, mask=mask)
        g = tl.load(x_ptr + idx * 2 + 1, mask=mask)
        g_silu = g * tl.sigmoid(g)
        output = f * g_silu

        tl.store(output_ptr + idx, output, mask=mask)
