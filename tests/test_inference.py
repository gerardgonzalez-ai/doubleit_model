import pytest
import torch
import time
from model import load_model, infer


def test_load_model():
    """
    Unit Test:
    Test the load_model function by checking if it returns a torch.jit.ScriptModule instance.
    """
    model_path = './model/doubleit_model.pt'
    model = load_model(model_path)
    assert isinstance(model, torch.jit.ScriptModule)


@pytest.mark.parametrize(
    "input_tensor, expected_output",
    [
        (torch.tensor([1, 2, 3, 4]), torch.tensor([2, 4, 6, 8])),
        (torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0, 0])),
        (torch.tensor([3, 6, 9, 12]), torch.tensor([6, 12, 18, 24])),
    ]
)
def test_infer(input_tensor: torch.Tensor, expected_output: torch.Tensor):
    """
    Parametrized Test:
    Parametrized test for the infer function, checking against expected outputs.
    """
    model_path = './model/doubleit_model.pt'
    model = load_model(model_path)
    result = infer(model, input_tensor)
    assert torch.equal(result, expected_output)


def test_infer_with_empty_tensor():
    """
    Edge Case Test:
    Test the infer function with an empty tensor, expecting an empty tensor as output.
    """
    model_path = './model/doubleit_model.pt'
    model = load_model(model_path)
    empty_tensor = torch.tensor([])
    result = infer(model, empty_tensor)
    assert torch.equal(result, torch.tensor([]))


def test_infer_with_negative_values():
    """
    Edge Case Test:
    Test the infer function with negative values, expecting the output to be negative as well.
    """
    model_path = './model/doubleit_model.pt'
    model = load_model(model_path)
    negative_tensor = torch.tensor([-1, -2, -3, -4])
    result = infer(model, negative_tensor)
    assert torch.equal(result, torch.tensor([-2, -4, -6, -8]))


def test_infer_with_different_data_types():
    """
    Data Types Test:
    Test the infer function with floating point values, expecting the output to be floating point as well.
    """
    model_path = './model/doubleit_model.pt'
    model = load_model(model_path)
    float_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = infer(model, float_tensor)
    assert torch.equal(result, torch.tensor([2.0, 4.0, 6.0, 8.0]))


def test_inference_speed():
    """
    Performance Test:
    Test the speed of inference, expecting it to complete in less than 0.1 seconds.
    """
    model_path = './model/doubleit_model.pt'
    model = load_model(model_path)
    sample_tensor = torch.tensor([1, 2, 3, 4])
    
    start_time = time.time()
    infer(model, sample_tensor)
    duration = time.time() - start_time
    
    assert duration < 0.1
