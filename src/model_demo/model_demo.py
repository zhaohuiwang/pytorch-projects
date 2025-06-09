
import logging
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Iterator, Any, Tuple
import numpy.typing as npt
from src.model_demo.utils import LinearRegressionModel, load_data, infer_evaluate_model, setup_logger, device
from src.model_demo.config import MetadataConfigSchema

#data_dir = "data/model_demo"
#data_fname = "data_tensors.pt"
#model_dir = "models/model_demo"
#model_fname = "demo_model_weights.pth"

config = MetadataConfigSchema()
data_dir = config.data.data_dir
data_fname = config.data.data_fname
model_dir = config.data.model_dir
model_fname = config.data.model_fname

learning_rate = config.model.learning_rate
batch_size = config.model.batch_size
epochs = config.model.epochs

## Initialize logger
logger = setup_logger(log_file=f'{data_dir}/model_logfile.log')

logger.info(f"Using {device} device")

## Prepare the dataset - Load tensor
try:
    tensors_dict = torch.load(Path(data_dir) / data_fname)
except FileNotFoundError:
    logger.error('Data or path not fund!')

X_train = tensors_dict['X_train']
X_test = tensors_dict['X_test']
y_train = tensors_dict['y_train']
y_test = tensors_dict['y_test']


## Model training
# Step 1: Create model class
# in the utils.py

# Step 2: Instantiate model class
input_dim = X_train.shape[-1]
output_dim = y_train.shape[-1]
model = LinearRegressionModel(input_dim, output_dim)
model.to(device)

# Step 3: Instantiate Loss class and Optimizer class
# learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Step 4: Train the model
# batch_size = 100   # batch_size should be a positive integer value
# epochs = 100

# Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
data_iter = load_data((X_train, y_train), batch_size)

for epoch in range(epochs):
    epoch += 1 # Logging starts at 1 instead of 0
    for X, y in data_iter:
       X = X.to(device)
       y = y.to(device)
       
       optimizer.zero_grad() # Clear gradients w.r.t. parameters
       outputs = model(X) # Forward to get output
       loss = criterion(outputs, y) # Calculate Loss

       # loss_list.append(loss.item())
       # epoch_list.append(epoch)
       print('epoch {}, loss {}'.format(epoch, loss.item())) # Logging
       
       loss.backward() # Getting gradients w.r.t. parameters
       optimizer.step() # Updating parameters

# step 5: Persist the trained model 
torch.save(model.state_dict(), Path(model_dir) / model_fname)

logger.info("Model training accomplished!")
logger.info(f"Model is saved as {Path(model_dir) / model_fname}")

## Model inference
if __name__ == "__main__":

    logger.info("Starting the model-demo inference")

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    # Process the test data set
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    test_data_iter = load_data((X_test, y_test), batch_size, is_train=False)

    # Perform inference
    predictions, avg_loss = infer_evaluate_model(model, test_data_iter, criterion)

    # Convert to NumPy and save
    numpy_predictions = predictions.cpu().numpy()

    np.save(Path(data_dir) / 'predictions.npy', numpy_predictions)
    np.savetxt(Path(data_dir) / 'predictions.csv', numpy_predictions, delimiter=',', fmt='%d')
    logger.info(f"Inference result is saved in directory: {data_dir}")


"""
To save the trained model for later use

# PyTorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save method
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.

modelX = nn.Linear(1, 1)
modelX.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# Using weights_only=True is considered a best practice when loading weights.
modelX.to(device)

# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results. If you wish to resuming training, call model.train() to set these layers to training mode.
modelX.eval()


# Load .npy file
loaded_npy = np.load('mnist_predictions.npy')
print("Loaded .npy:", loaded_npy[:5])

# Load .csv file
loaded_csv = np.loadtxt('mnist_predictions.csv', delimiter=',')
print("Loaded .csv:", loaded_csv[:5])

"""
