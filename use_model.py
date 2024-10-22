import numpy as np


import torch
import torch.nn as nn

# Define your model architecture as done previously
from train import SimpleNN

def load_model(file_name: str = 'point_distance_model.pth') -> SimpleNN:
    model = SimpleNN()  
    model.load_state_dict(torch.load(file_name,weights_only=True))
    model.eval()  
    return model

def prepare_input_data(num_points: int = 10) -> torch.Tensor:
    input_data = np.random.uniform(-1, 1, (num_points, 3)).astype(np.float32)
    return torch.tensor(input_data)

def make_predictions(model: SimpleNN, input_tensor: torch.Tensor):
    with torch.no_grad():  
        predictions = model(input_tensor)
    return predictions

def main():
    model = load_model()  # Load the trained model
    input_tensor = prepare_input_data(10)  # Prepare input data
    predictions = make_predictions(model, input_tensor)  # Make predictions
    print("Predictions:\n", predictions)

if __name__ == "__main__":
    main()

