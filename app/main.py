# Importing necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model.inference import load_model, infer

# Initializing FastAPI app
app = FastAPI()

# Loading the model
model_path = './model/doubleit_model.pt'
model = load_model(model_path)

class InputData(BaseModel):
    """
    Schema for the input data.
    Defines a list of floats as the expected input.
    """
    numbers: list[float]

@app.post("/predict/")
async def predict(input_data: InputData):
    """
    Endpoint for model inference.
    Takes a list of numbers, converts to tensor, performs inference, 
    and returns the result as a list of numbers.

    Parameters:
        input_data (InputData): Input data with a list of numbers.

    Returns:
        dict: A dictionary with the result key and the inference output as a list.
    """
    # Converting input data to tensor
    tensor = torch.tensor(input_data.numbers)
    
    # Performing inference
    result = infer(model, tensor)
    
    # Returning result
    return {"result": result.tolist()}
