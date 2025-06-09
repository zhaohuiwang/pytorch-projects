from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Iterator, Any, Tuple
import numpy.typing as npt
from src.model_demo.utils import synthesize_data, norm, setup_logger
from src.model_demo.config import MetadataConfigSchema

# data_dir = "data/model_demo"
# data_fname = "data_tensors.pt"
# train_size = 0.8

config = MetadataConfigSchema()
data_dir = config.data.data_dir
data_fname = config.data.data_fname
train_size = config.model.train_size

## Initialize logger
logger = setup_logger(log_file=f'{data_dir}/data_logfile.log')

## Setup - data preparation

# for synthesizing data following y = xW^T + bias
true_w = torch.tensor([2., -3.])
true_b = torch.tensor(4.)

X, y = synthesize_data(true_w, true_b, 1000)

size = int(X.shape[-2]*train_size)

# np.random.choice() generates a random sample from a given 1D array. here is is sampling size/total_size 
index = np.random.choice(X.shape[-2], size=size, replace=False) 

# Prepare the traing set. Note the synthetic data are torch.Tensors. Here it is transform into NumpyArray first for norm operation then reverse back to Tensor.
X_train = torch.from_numpy(norm(X[index].numpy()))
y_train = y[index]

## Prepare the test set.
X_test = torch.from_numpy(norm(np.delete(X, index, axis=0).numpy()))
y_test = np.delete(y, index, axis=0)


if __name__ == "__main__":
  
  # Store tensors in a dictionary
  tensors_dict = {
      'X_train': X_train,
      'X_test': X_test,
      'y_train': y_train,
      'y_test': y_test
      }

  # Save to a file - A common PyTorch convention is to save tensors using .pt file extension.
  torch.save(tensors_dict, Path(data_dir) / data_fname) 

  logger.info(f"Data is saved as: {Path(data_dir) / data_fname}")


# within the project director execute `python src/model_demo/data_prep.py` 




"""
Alternatively
# Store tensors in a list
tensors_dict = {
    'train': [X_train, y_train],
    'test': [X_test, y_test]
    }
# to access: X_train = tensors_dict['train'][0]; y_train = tensors_dict['train'][1]

# Stack tensors into a single tensor (adds a new dimension)
tensors_dict = {
    'train': torch.stack([X_train, y_train]),
    'test': torch.stack([X_test, y_test])
    }
# to access, same as the list: X_train = tensors_dict['train'][0]; y_train = tensors_dict['train'][1]

# Load tensor
tensors_dict = torch.load(saved_path)
"""