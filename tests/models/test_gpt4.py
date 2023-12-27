# test_gpt4.py
import torch
from zeta.models import GPT4


# Test the creation of a GPT4 model with the default parameters.
def test_default_model_creation():
    default_model = GPT4()
    assert isinstance(default_model, GPT4)


# Check the use_abs_pos_emb parameter.
def test_use_abs_pos_emb_parameter():
    model = GPT4(use_abs_pos_emb=True)
    assert model.use_abs_pos_emb is True


# Check the forward function.
def test_forward_function():
    model = GPT4()
    text_tokens = torch.tensor(
        [[2, 5, 9], [4, 1, 8]]
    )  # Add more test cases here.
    result = model.forward(text_tokens)
    assert result.size() == (2,)  # Replace with the expected result size.


# Add more tests for different parameters, edge cases, and error conditions.
# Also add tests for other methods present in the class, if any.
