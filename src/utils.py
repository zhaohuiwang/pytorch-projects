



from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def accuracy(x, y, model):
   """
    A function that calculates the accuracy of a given dataset. In the preceding code, we explicitly mention that we don't need to calculate the gradient by providing @torch.no_grad() and calculating the prediction values, by feed-forwarding input through the model.
   """
   model.eval() # set the dropout and batch normalization layers
   # get the prediction matrix for a tensor of `x` images
   prediction = model(x)
   # compute if the location of maximum in each row coincides with ground truth
   # prediction.max(dim) returns the maximum values and their indices along the specified dimension.
   max_values, argmaxes = prediction.max(-1) # -1 specifies the last dimension
   is_correct = argmaxes == y
   return is_correct.cpu().numpy().tolist()

def accuracy(x, y, model):
   """
   A function that calculates the accuracy of a given dataset.
   """
   model.eval()
   with torch.no_grad():
      prediction = model(x)[0]
    max_values, argmaxes = prediction.max(-1) # -1 specifies the last dimension
   is_correct = argmaxes == y
   return is_correct.cpu().numpy().tolist()

