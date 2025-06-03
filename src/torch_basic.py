""""
https://github.com/PacktPublishing/
https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E

PyTorch is based on Torch, a framework for doing fast computation that is written in C. Torch has a Lua (means "Moon" in Portuguese) wrapper for constructing models.

One of the main special features of PyTorch is that it adds a C++ module for autodifferentiation to the Torch backend using torch.autograd engine. 
By default, PyTorch uses eager mode computation. Same as the Keras


PyTorch Ecosystem
fast.ai: An API that makes it straightforward to build models quickly.
TorchServe: An open-source model server developed in collaboration between AWS and Facebook.
TorchElastic: A framework for training deep neural networks at scale using Kubernetes.
PyTorch Hub: An active community for sharing and extending cutting-edge models.
TorchVison: A library dedicated to computer vision tasks that offers datasets, model architectures, and common image transformations.
TorchAudio: A library for audio processing and audio manipulation utilities.
TorchText: A natural language processing library that provides data processing utilities and popular datasets for the NLP field.

"""

import numpy as np
import torch

def tanh(x):
 return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu(x):
 return np.where(x>0,x,0)

def linear(x):
 return x

def softmax(x):
 return np.exp(x)/np.sum(np.exp(x))

def mse(p, y):
 return np.mean(np.square(p - y))

def mae(p, y):
 return np.mean(np.abs(p-y))

def binary_cross_entropy(p, y):
 return -np.mean((y*np.log(p)+(1-y)*np.log(1-p)))

def categorical_cross_entropy(p, y):
 return -np.mean(np.log(p[np.arange(len(y)),y]))


# Converting NumPy objects to tensors is baked into PyTorch’s core data structures. you can easily switch back and forth between torch.Tensor objects and numpy.array objects using torch.from_numpy() and Tensor.numpy() methods.
import torch
import numpy as np

x = np.array([[2., 4., 6.]])
y = np.array([[1.], [3.], [5.]])

m = torch.mul(torch.from_numpy(x), torch.from_numpy(y))

m.numpy() 
# exact same method for both PyTorch and TensorFlow whereas DataFrame.to_numpy()



# With PyTorch, requires_grad=True parameter signals to torch.autograd engine that every operation on them should be tracked. (with TensorFlow we need the tf.GradientTape API)
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
L = 3*a**3 - b**2
# We can call .backward() on the loss function (L) of a and b, autograd calculates gradients of the L w.r.t parameters and store them in the respective's tensors' .grad attribute. For example,
external_grad = torch.tensor([1., 1.])
L.backward(gradient=external_grad)
# the gradient parameter specifies the gradient of the function being differentiated w.r.t. self. This argument can be omitted if self is a scalar. here we have a and b.
print(a.grad); print(9*a**2)
print(b.grad); print(-2*b)



import torch

# Import pprint, module we use for making our print statements prettier
import pprint
pp = pprint.PrettyPrinter()

# Create the inputs
input = torch.ones(2,3,4)

# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None) y = xA^T + b.

model = torch.nn.Linear(in_features=4, out_features=2) 
# the weights and bias are initalized randomly from U(-sqrt{k}, sqrt{k}) where k=1/ in_features. The size of the weight matrix is out_features x in_features, and the size of the bias vector is out_features.

linear_output = model(input)
linear_output.shape



######### Linear Regression #########
# from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import pandas as pd


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define a function to generate noisy data
def synthesize_data(w, b, sample_size):
  """ Generate y = xW^T + bias + noise """
  X = torch.normal(0, 1, (sample_size, len(w)))
  y = torch.matmul(X, w) + b# add noise
  y += torch.normal(0, 0.01, y.shape)
  
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2., -3.])
true_b = 4.
features, labels = synthesize_data(true_w, true_b, 1000)

def load_data(data_arrays, batch_size, shuffle=True):
 """
 Construct a PyTorch data iterator.
 torch.utils.data.TensorDataset(*tensors) wraps tensors (samples and their corresponding labels). Each sample will be retrieved by indexing tensors along the first dimension.
 torch.utils.data.DataLoader() provides an iterable over the given dataset.
 """
 dataset = TensorDataset(*data_arrays)
 return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
 
batch_size = 10
data_iter = load_data((features, labels), batch_size)
# next(iter(data_iter))
len(data_iter) # sample_size / batch_size = 1000 / 10 = 100

# Create a single layer feed-forward network with 2 inputs and 1 outputs.
model = nn.Linear(2, 1).to(device)
# define a loss function: mean squared error
criterion = nn.MSELoss()
# define a optimization method: stochastic gradient descent 
lr=0.03
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Initialize model params
# the default weights and bias are initalized randomly from U(-sqrt{k}, sqrt{k}) where k=1/ in_features.
model.weight.data.normal_(0, 0.01)
model.bias.data.fill_(0)

# train for n epochs, iteratively using minibatch of the size defined by batch_size vriable
num_epochs = 5
losslist = []
epochlist = []

# When you perform backpropagation, the gradients of the loss with respect to the model's parameters are calculated and stored. If you don't zero out the gradients before the next iteration, the gradients from the previous iteration will be added to the current gradients, leading to incorrect updates.

for epoch in range(num_epochs):
 for X, y in data_iter:
  
  X, y = X.to(device), y.to(device)
  # forward pass
  y_out = model(X)
  l_t = criterion(y_out, y)
  # zero out the gradients 
  optimizer.zero_grad() 

  # backpropagation
  l_t.backward()
  # Update the parameters
  optimizer.step()

  yv_out = model(features)
  l_v = criterion(yv_out, labels)
  print(f'epoch {epoch + 1}, loss {l_v:f}')

  losslist.append(l_v.item())
  epochlist.append(epoch)

result_df = pd.DataFrame({'epoch':epochlist, 'loss':losslist})

# Results
w = model.weight#.tolist()
print('Error in estimating weights:', true_w - w.reshape(true_w.shape))
b = model.bias#.item()
print('Error in estimating bias:', true_b - b)

# Tensor.tolist() returns the tensor as a (nested) list. For scalars, a standard Python number is returned, just like with Tensors.item(). Tensors are automatically moved to the CPU first if necessary. tensor.data attribute was previously used to access the underlying storage of a tensor. However, it's now considered deprecated. Directly modifying tensor.data is generally discouraged as it can lead to unexpected behavior in PyTorch's autograd system.

summary(model) # imilar to Tensorflow's model.summary()


import torch
import torch.nn as nn
import numpy as np

# Setup - data preparation
# Define a function to generate noisy data
def synthesize_data(w, b, sample_size):
  """ Generate y = xW^T + bias + noise """
  X = torch.normal(10, 3, (sample_size, len(w)))
  y = torch.matmul(X, w) + b# add noise
  y += torch.normal(0, 0.01, y.shape)
  
  return X, y.reshape((-1, 1))

def norm(x):
    """ normalize the original data values """
    return (x - np.mean(x)) / np.std(x)

def load_data(tensors, batch_size, is_train=True):
   """ Construct a PyTorch data iterator."""
   dataset = torch.utils.data.TensorDataset(*tensors)
   return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)
 
true_w = torch.tensor([2., -3.])
true_b = 4.

train_size = 0.8

X, y = synthesize_data(true_w, true_b, 1000)
size = int(X.shape[-2]*train_size)
index = np.random.choice(X.shape[-2], size=size, replace=False) 

# Prepare the traing set. Note the synthetic data are torch.Tensors. Here it is transform into NumpyArray first for norm operation then reverse back to Tensor.
X_train = torch.from_numpy(norm(X[index].numpy()))
y_train = y[index]
# Prepare the test set.
X_test = torch.from_numpy(norm(np.delete(X, index, axis=0).numpy()))
y_test = np.delete(y, index, axis=0)

# Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
data_iter = load_data((X_train, y_train), batch_size)



# Use GPU when available, the default is CPU 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Step 1: Create model class
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        out = self.linear(x)
        return out

# Step 2: Instantiate model class
input_dim = X_train.shape[-1]
output_dim = y_train.shape[-1]
model = LinearRegressionModel(input_dim, output_dim)
model.to(device)

# Step 3: Instantiate Loss class and Optimizer class
criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Step 4: Train the model
batch_size = 10
epochs = 100

# loss_list = []
# epoch_list = []
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



###########

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchinfo import summary
import pandas as pd


# It is necessary to have both the model, and the data on the same device, either CPU or GPU, for the model to process data.

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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values (the minibatch dimension (at dim=0) is maintained).
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # apply a linear transformation on the input using its stored weights and biases
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

"""
Inside the training loop, optimization happens in three steps:

optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

Call loss.backward() to backpropagate the prediction loss. PyTorch deposits the gradients of the loss w.r.t. each parameter.

Call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
"""
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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


learning_rate = 1e-3
batch_size = 64
epochs = 10

# Initialize model, and move it to the device    
model = NeuralNetwork()#.to(device)

print(summary(model))

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
   print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
  print(var_name, "\t", optimizer.state_dict()[var_name])




# PyTorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save method
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.
modelX = NeuralNetwork()
# Using weights_only=True is considered a best practice when loading weights.
modelX.load_state_dict(torch.load('model_weights.pth', weights_only=True))
modelX.to(device)
# # Make sure to call input = input.to(device) on any input tensors that you feed to the model

# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results. If you wish to resuming training, call model.train() to set these layers to training mode.
modelX.eval()





X = torch.rand(1, 28, 28, device=device)
logits = model(X)

pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print(f"Predicted class: {y_pred}")
# Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the nn.Softmax module.

# 3 images of size 28x28
input_image = torch.rand(3,28,28)
flatten = nn.Flatten()
flat_image = flatten(input_image)
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
hidden1 = nn.ReLU()(hidden1)



# https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html

import torch
import matplotlib.pyplot as plt
import numpy as np

z = torch.zeros(5, 3)
print(z)
print(z.dtype)

i = torch.ones((5, 3), dtype=torch.int16)
print(i)


torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed

# Tensors of similar shapes may be added, multiplied, etc. Operations with scalars are distributed over the tensor
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos       # addition allowed because shapes are similar
print(threes)              # tensors are added element-wise
print(threes.shape)        # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2


r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r)) # torch.std(r); torch.mean(r)
print('\nMaximum value of r:')
print(torch.max(r))


import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

net = LeNet()
print(net)                         # what does the object tell us about itself?

input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # we don't call forward() directly
print('\nRaw output:')
print(output)
print(output.shape)

"""
First, we instantiate the LeNet class, and we print the net object. A subclass of torch.nn.Module will report the layers it has created and their shapes and parameters. This can provide a handy overview of a model if you want to get the gist of its processing.

Below that, we create a dummy input representing a 32x32 image with 1 color channel. Normally, you would load an image tile and convert it to a tensor of this shape.

You may have noticed an extra dimension to our tensor - the batch dimension. PyTorch models assume they are working on batches of data - for example, a batch of 16 of our image tiles would have the shape (16, 1, 32, 32). Since we're only using one image, we create a batch of 1 with shape (1, 1, 32, 32).

We ask the model for an inference by calling it like a function: net(input). The output of this call represents the model's confidence that the input represents a particular digit. (Since this instance of the model hasn't learned anything yet, we shouldn't expect to see any signal in the output.) Looking at the shape of output, we can see that it also has a batch dimension, the size of which should always match the input batch dimension. If we had passed in an input batch of 16 instances, output would have a shape of (16, 10).
"""



# Datasets and Dataloaders
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

""""
transforms.ToTensor() converts images loaded by Pillow into PyTorch tensors.

transforms.Normalize() adjusts the values of the tensor so that their average is zero and their standard deviation is 1.0. Most activation functions have their strongest gradients around x = 0, so centering our data there can speed learning. The values passed to the transform are the means (first tuple) and the standard deviations (second tuple) of the rgb values of the images in the dataset.
There are many more transforms available, including cropping, centering, rotation, and reflection.

# download CIFAR10 dataset which is a set of 32x32 color image tiles representing 10 classes of objects: 6 of animals (bird, cat, deer, dog, frog, horse) and 4 of vehicles (airplane, automobile, ship, truck)

When we instantiate our dataset, we need to tell it a few things:
The filesystem path to where we want the data to go.
Whether or not we are using this set for training; most datasets will be split into training and test subsets.
Whether we would like to download the dataset if we haven’t already.
The transformations we want to apply to the data.

"""

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True, transform=transform
    )

# Once your dataset is ready, you can give it to the DataLoader. we’ve asked a DataLoader to give us batches of 4 images from trainset, randomizing their order (shuffle=True), and we told it to spin up two workers to load data from disk.
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
shuffle=True, num_workers=2
)

# good practice to visualize the batches your DataLoader serves
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# Training Your PyTorch Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


"""""
Build the Neural Network

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
    def __init__(self):
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

    def forward(self, x):
        x = self.flatten(x)
        # for an input (3,28,28) the output will be (3, 784) with the minibatch dimension maintained after converting a 2D into a contiguous array
        logits = self.layer_stack(x)
        return logits
    
# create an instance of NeuralNetwork, and move it to the device
model = NeuralNetwork().to(device)

# print the model structure
print(f"Model structure:\n {model}")


for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# To use the model, we pass it the input data. This executes the model's forward, along with some background operations. Do not call model.forward() directly!
X = torch.rand(1, 28, 28, device=device) # 1 image of size 28x28
logits = model(X)

""" 
Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of n (10 in this example) raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the nn.Softmax module.
The logits raw values are in [-infty, infty], nn.Softmax module applies the normalized exponential function to convert the logits values into a probability distribution of K possible outcomes in the range of [0, 1].
The dim parameter indicates the dimension along which the values must sum to 1.
"""

# predicted probabilities
pred_probab = nn.Softmax(dim=1)(logits)
# predicted class
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# Compute squared euclidean distance
import torch
a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
torch.cdist(a, b, p=2)
