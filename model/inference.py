import torch


def load_model(model_path: str) -> torch.jit.ScriptModule:
    """
    Load a TorchScript module from the specified path.
    
    Parameters:
        model_path (str): The path to the model file.
        
    Returns:
        torch.jit.ScriptModule: The loaded model.
    """
    return torch.jit.load(model_path)


def infer(model: torch.jit.ScriptModule, tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform inference on the given tensor using the specified model.
    
    Parameters:
        model (torch.jit.ScriptModule): The model to use for inference.
        tensor (torch.Tensor): The input tensor.
        
    Returns:
        torch.Tensor: The output tensor.
    """
    return model(tensor)


if __name__ == "__main__":
    model_path = './doubleit_model.pt'
    model = load_model(model_path)
    sample_tensor = torch.tensor([1, 2, 3, 4])
    result = infer(model, sample_tensor)
    print(result)  # <- tensor([2, 4, 6, 8])
