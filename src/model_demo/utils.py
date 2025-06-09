

import logging
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel
from typing import Iterator, Any, Tuple, Union


def setup_logger(log_file:str='app.log', log_level=logging.DEBUG):
    """
    Create a centralized logger configuration.
    List of logging levels: DEBUG (10) >INFO (20) > WARNING (30) > ERROR (40) > CRITICAL (50)
    """
    # Create logger
    logger = logging.getLogger('MyAppLogger')
    logger.setLevel(log_level)
    
    # Prevent adding handlers if logger is already configured
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation (max 5MB, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


device = lambda: ("cuda" if torch.cuda.is_available() else 
                  "mps"  if torch.backends.mps.is_available() else 
                  "cpu"
                  )

# Define a function to generate noisy data
def synthesize_data(w: torch.Tensor, b: torch.Tensor, sample_size) -> Tuple[torch.Tensor, torch.Tensor]:
  """ Generate y = xW^T + bias or noise """
  X = torch.normal(10, 3, (sample_size, len(w)))
  y = torch.matmul(X, w) + b # adding noise
  y += torch.normal(0, 0.01, y.shape)
  
  return X, y.reshape((-1, 1))

def norm(x:npt.NDArray) -> npt.NDArray:
    """ normalize the original data values """
    return (x - np.mean(x)) / np.std(x)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        out = self.linear(x)
        return out

# Define the request body format for predictions
class PredictionFeatures(BaseModel):
    feature_X_1: Union[int, float]
    feature_X_2: Union[int, float]
    
       
def load_data(tensors: torch.Tensor, batch_size:torch.Tensor, is_train: bool=True) -> Iterator[Any]:
   """ Construct a PyTorch data iterator."""
   dataset = torch.utils.data.TensorDataset(*tensors)
   return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=2, pin_memory=True)


# Function for inference and loss calculation
def infer_evaluate_model(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, float]:
    model.eval()  # Set model to evaluation mode
    model.to(device)

    predictions = torch.tensor([])

    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradients for inference
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass (inference)
            outputs = model(inputs)
            # Append using torch.cat
            
            predictions = torch.cat((predictions, outputs))
            # Alternatively, use torch.hstack (same for 1D tensors)
            # predictions = torch.hstack((predictions, outputs))

            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    # Calculate average loss
    avg_loss = total_loss / total_samples
    return predictions, avg_loss

# Function for inference 
def infer_model(model, inputs, device='cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:

    model.eval()  # Set model to evaluation mode
    model.to(device)

    with torch.no_grad():  # Disable gradients for inference
        inputs  = inputs.type(torch.float).to(device) 
        # Forward pass (inference)
        outputs = model(inputs)
    
    return outputs