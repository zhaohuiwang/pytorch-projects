


###############  Build the Neural Network ############### 
"""
https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

Neural networks comprise of layers/modules that perform operations on data. The torch.nn namespace provides all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module. A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.

a neural network to classify images in the FashionMNIST dataset

Get Device for Training
We want to be able to train our model on a hardware accelerator like the GPU or MPS (Mac M1/M2/M3), if available. Let's check to see if torch.cuda or torch.backends.mps are available, otherwise we use the CPU.

Define the Class
We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__. Every nn.Module subclass implements the operations on input data in the forward method.

CLASS
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
Applies an affine linear transformation to the incoming data: 
y = xA^T + b
weight A of shape (out_features, in_features) and bias b of shape (out_features) are learnable variables, when bias=True.


Model Parameters
Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. Subclassing nn.Module automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model's parameters() or named_parameters() methods.

"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# Get Device for Training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define the Class
class NeuralNetwork(nn.Module):
    # It is mandatory to inherit from nn.Module when creating a model architecture, as it is the base class for all neural network modules
    def __init__(self):
        # to take advantage of all the pre-built functionalities in nn.Module
        super().__init__()
        self.flatten = nn.Flatten()
        # Construct a stack of layers
        self.layer_stack = nn.Sequential(
           # nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined.
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            # nn.ReLU() is a Non-linear activation to introduce nonlinearity
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
            # the linear applies a linear transformation on the input using its stored weights and bias and output size is (minibatch, out_features)
        )
    # define the forward-propagation of the network - input > output > input...
    def forward(self, x):
        x = self.flatten(x)
        # for an input (3,28,28) the output will be (3, 784) with the minibatch dimension maintained after converting a 2D into a contiguous array
        logits = self.layer_stack(x)
        return logits
    
# create an instance of NeuralNetwork, and move/register it to the device
model = NeuralNetwork().to(device)

opt_mod = torch.compile(model)
"""
torch.compile allows PyTorch to automatically optimize your models instances and Arbitrary Python functions for better performance by applying just-in-time (JIT) compilation(comparing to Ahead-of-Time compilation) and various graph optimizations, making use of multiple backends (e.g., TorchDynamo, XLA, TorchInductor).
Optimized a function 
1. Arbitrary Python functions can be optimized by passing the callable to torch.compile. 
2. Alternatively, we can decorate the function with @torch.compile

Optimized a model instance: opt_mod = torch.compile(model)

To disable some function from being compiled
@torch.compiler.disable(recursive=False | True)


"""

# print the model structure
print(f"Model structure:\n {model}")


for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

for _ in model.named_parameters():
    print(_)
""" 
Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of n (10 in this example) raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the nn.Softmax module.
The logits raw values are in [-infty, infty], nn.Softmax module applies the normalized exponential function to convert the logits values into a probability distribution of K possible outcomes in the range of [0, 1].
The dim parameter indicates the dimension along which the values must sum to 1.

# To use the model, we pass it the input data. This executes the model's forward, along with some background operations. Do not call model.forward() directly!
X = torch.rand(1, 28, 28, device=device) # 1 image of size 28x28
logits = model(X)


# predicted probabilities
pred_probab = nn.Softmax(dim=1)(logits)
# predicted class
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
"""



############### Optimizing Model Parameters ############### 
"""
** Hyperparameters
1. Number of Epochs - the number times to iterate over the dataset
2. Batch Size - the number of data samples propagated through the network before the parameters are updated
3. Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
Number of batchs or iterations per epoch = dataset size / batch size

** Each epoch consists of two main parts:
1. The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
2. The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.

** Loss Function - measures the degree of dissimilarity of obtained result to the target value, and it is the function that we want to minimize during training.
Common loss functions include 
nn.MSELoss() (Mean Square Error) for regression tasks.
nn.NLLLoss() (Negative Log Likelihood) for classification. 
nn.CrossEntropyLoss() combines nn.LogSoftmax and nn.NLLLoss.

** Optimizer - the process of adjusting model parameters to reduce model error in each training step. 
SGD, ADAM, RMSProp, ...

** Inside the training loop, optimization happens in three steps:
1. Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
2. Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
3. Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
"""

"""
Within a training loop, we iterate over the following steps
1. PyTorch initialize parameters by default, so even in the 1st, we run model prediction with code: pred = model(X)
2. calculate the loss with code: loss = loss_fn(pred, y)
3. carry out backproparation with code: loss.backward()
4. update the parameters (on the top of initiation or previous step) with code: optimizer.step()
5. zero out the gradient, and ready for the next iteration
6. return to step 1
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        ## feed-forward and compute prediction error
        pred = model(X) # prediction by feed-forwarding input through the model.
        loss = loss_fn(pred, y) # compute loss

        ## Backpropagation
        loss.backward() # calculates the gradients of the loss w.r.t. the weights
        optimizer.step() # update the weights according to the gradients computed and the learning rate (determine the how big a step)
        optimizer.zero_grad() # zero out the gradients after the weight update step and before the next iteration, default is adding up.


        ## there are dataset size / batch size iterations per epoch, we only print every 100th
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




# prepare data - and download to the directory specified by the root parameter
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)



learning_rate = 1e-3
batch_size = 64 # Typical values are powers of 2 (e.g., 32, 64, 128)
epochs = 10

# DataLoader wraps an iterable around the Dataset to enable easy access to the samples. After we iterate over all batches the data is shuffled as specified by shuffle=True.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# Initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning rate hyperparameter.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


############### Save and Load the Model ############### 
"""
Two appraoches:
Model Weights only
PyTorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save() method.

To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.

Mpodel structure and Model Weights

"""

# BEST PRACTICE: save the model weights and load the model
torch.save(model.state_dict(), 'model_weights.pth') # save the model

model = NeuralNetwork().to(device) # create a instance of the same model first
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# Load the model. Using weights_only=True is considered a best practice

# set the dropout and batch normalization layers to evaluation mode
model.eval()


# LEGACY: Saving and Loading Models with shapes (both the structure of network and parameter weights)
torch.save(model, 'model.pth')
model = torch.load('model.pth', weights_only=False)

############### Model Prediction ############### 

test_data.to(device)
model


