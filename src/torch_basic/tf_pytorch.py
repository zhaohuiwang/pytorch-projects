

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 
import numpy as np

# Tensors can be created directly from data, Numpy arrays, another tensor or with random / constant values.
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array) # .numpy() method to reverse back to numpy

x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype

shape = (2,3,)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
rand_tensor = torch.rand(3,4) # from a uniform distribution on [0,1)
rand_tensor = torch.rand(size= (3,4)) # same as the above

"""

torch.tensor(data, requires_grad=False, ...) constructs a tensor with no autograd hisory by copying data. By default, requires_grad=False

tf.constant(value, dtype=None, shape=None, name='Const') Creates a constant tensor from a tensor-like object.
tf.Variable(initial_value=None,trainable=None,name=None,dtype=None,shape=None,...) A variable maintains shared, persistent state manipulated by a program.

data = [[1, 2],[3, 4]]
x_data = tf.constant(data)
x_data
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]], dtype=int32)>

x_np = tf.convert_to_tensor(np_array)
x_np
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [3, 4]], dtype=int32)>

x_ones = tf.ones_like(x_data)
x_ones
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 1],
       [1, 1]], dtype=int32)>

       torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, ...)
size (int...) – a sequence of integers defining the shape of the output tensor. Can be a variable number of arguments or a collection like a list or tuple.


Note: the difference between tf and pytorch in the shape/size parameters. 

torch.zeros(*size, *, ...)  #
torch.randint(low: int=0, high: int, size: tuple,...) 
torch.rand(*size *, ...)    # from between 0 and 1
torch.randn(*size, *, ...)   # from normal distribution N(0,1)

tf.ones(shape, dtype=tf.dtypes.float32, name=None,layout=None)
# shape	A list of integers, a tuple of integers, or a 1-D Tensor of type int32.
tf.zeros(shape, dtype=tf.dtypes.float32, name=None,layout=None)
tf.keras.random.randint(shape, minval, maxval, dtype='int32', seed=None)

tf.random.uniform(shape,minval=0,maxval=None,dtype=tf.dtypes.float32,seed=None,name=None)
tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)

"""

# tensor attributes
rand_tensor = torch.rand(3, 4)
rand_tensor.shape # torch.Size([3, 4])
rand_tensor.size() # torch.Size([3, 4])
rand_tensor.dtype # torch.float32
rand_tensor.device # device(type='cpu')
rand_tensor.ndim # 2

"""
# Tensor Flow
t1 = tf.constant([[1,2,3,4],[5,6,7,8]])
t1.shape    # TensorShape([2, 4])
t1.ndim     # 2
t1.dtype    # tf.int32
tf.rank(t1) # <tf.Tensor: shape=(), dtype=int32, numpy=2>
tf.shape(t1) # <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 4], dtype=int32)>

"""
# The default device is initially cpu. 
torch.get_default_device() # device(type='cpu')
# do not run this unless you have GPU on the host
torch.set_default_device('cuda')  # current device is 0
torch.set_default_device('cpu')
# torch.get_default_device()
torch.cuda.set_device('cuda:1')  # current device is 1
# torch.get_default_device()

cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

x = torch.tensor([1., 2.], device=cuda0) # device(type='cuda', index=0)
y = torch.tensor([1., 2.]).cuda() # device(type='cuda', index=0)

with torch.cuda.device(1): # GPU 1
    a = torch.tensor([1., 2.], device=cuda) # allocates a tensor on GPU 1
    b = torch.tensor([1., 2.]).cuda() # transfers a tensor from CPU to GPU 1
    b2 = torch.tensor([1., 2.]).to(device=cuda) # transfer a tensor to GPU 1
    # device(type='cuda', index=1)
    c = a + b # c is on device(type='cuda', index=1)
    z = x + y # z is on device(type='cuda', index=0)

    # even within a context, you can specify the device
    # (or give a GPU index to the .cuda call)
    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)



# a context manager using the with statement 
with torch.device('cuda'):
    mod = torch.nn.Linear(20, 30)
    print(mod.weight.device)
    print(mod(torch.randn(128, 20)).device)

# Set it globally
torch.set_default_device('cuda')
mod = torch.nn.Linear(20, 30)
print(mod.weight.device)
print(mod(torch.randn(128, 20)).device)

# move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor_object.to("cuda")

"""
CUDA (Coupute unified Device Architecture) is a proprietary parallel computing platform and API that allows software to use certain types of GPUs for accelerated general-purpose processing, or general-propose computing on GPUs. CUDA was created by Nvidia in 2006.
# Tensor Flow
By default on GPU, fall back to CPU is GPU not available or the TensorFlow operation has no corresponding GPU implementation.
tf.device("/GPU:0") # Specifies the device to be used for ops created/executed in this context. 
with tf.device("/GPU:0"):
  operation execution on the specified device 

"""
# Standard numpy-like indexing and slicing
exam_tensor = torch.ones(4, 4)
exam_tensor[:,1] = 0
print(exam_tensor)

"""
TensorFlow can also subset using numpy-like indexing and slicing, not to mutate

tf.assign(tensor, new_values) function or method
tf.keras.backend.set_value(x, value) functions to modify the tensor
convert to numpy first(.numpy()), mutate it and then tf.convert_to_tensor()
...
"""

# join tensors
t1 = torch.cat([exam_tensor, exam_tensor, exam_tensor], dim=1)
# tf.concat(values, axis, name='concat') # values is a lost of tensor objects

# element-wise product
exam_tensor.mul(exam_tensor)
# Alternative syntax:
exam_tensor * exam_tensor

# matrix multiplication between two tensors
exam_tensor.matmul(exam_tensor.T)
# Alternative syntax:
exam_tensor @ exam_tensor.T

# in-place operation: operator with a _ suffix 
exam_tensor.add_(5)
exam_tensor.t_()

"""
tf.math.multiply(x, y, name=None) # returns an element-wise x * y.
x*y
tf.math.divide(x, y, name=None)
tf.matmul(a, b, ...) # multiply matrix a by matrix b
the @ operator is supported in TensorFlow it simply calls the tf.matmul() function.
"""


"""
two key components (functions) in model class:
Keras __init__() and call()
PyTorch: __init__() and forward() 
"""

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

torch.ones(4, 4).shape # same as 
torch.ones((4, 4)).shape
torch.ones((4,)).shape # same as 
torch.ones(4).shape


tensor[0]
tensor[:, 0]
tensor[..., -1]
tensor[:,1] = 0


t1 = torch.cat([tensor, tensor, tensor], dim=1)
# torch.cat(tensors, dim=0, *, out=None) # tensors (sequence of Tensors) 
# tf.concat(values, axis, name='concat') # values: A list of Tensor objects or a single Tensor.


t1 = torch.tensor([[1,2,3,4],[5,6,7,8]])
t1*10
tensor([[10, 20, 30, 40],
        [50, 60, 70, 80]])
t1.add(10)

t1 = tf.constant([[1,2,3,4],[5,6,7,8]])
t1*10
<tf.Tensor: shape=(2, 4), dtype=int32, numpy=
array([[10, 20, 30, 40],
       [50, 60, 70, 80]], dtype=int32)>
tf.add(t1, 10)


v = tf.Variable(1.)
v.assign(2.)
v.assign_add(0.5)

v = tf.Variable([1, 2, 3, 4, 5, 6])
# Assigns a new value to the variable.
v.assign([10, 20, 30, 40, 50, 60]) 
# Adds a value to this variable. shapes must be equal. 
v.assign_add([5, 5, 5, 5, 5, 6])
# Subtracts a value from this variable. shapes must be equal. 
v.assign_sub([10, 10, 10, 10, 10, 10])




# Initializer that generates tensors initialized to 0.
# Initializers allow you to pre-specify an initialization strategy, encoded in the Initializer object, without knowing the shape and dtype of the variable being initialized.
tf.zeros_initializer()
tf.ones_initializer()
tf.constant_initializer(value=0, support_partition=False)
tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
tf.random_uniform_initializer(minval=-0.05, maxval=0.05, seed=None)

IMAGE_SIZE = (28, 28)
input_shape=IMAGE_SIZE + (3,)
input_shape=(3,) + IMAGE_SIZE



import torch.nn as nn
import tensorflow as tf

# Dense Layers
dense_layer = nn.Linear(in_features=64, out_features=128)  
activation = nn.ReLU()    
dense_layer = tf.keras.layers.Dense(units=64, activation='relu')
# Convolutional Layers
conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)) 
activation = nn.ReLU()
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')

# Recurrent Layers
rnn_layer = nn.RNN(input_size=64, hidden_size=64, nonlinearity='tanh')
rnn_layer = tf.keras.layers.SimpleRNN(units=64, activation='tanh')

# Batch Normalization Layer
batch_norm_layer = nn.BatchNorm2d(num_features=64)
batch_norm_layer = tf.keras.layers.BatchNormalization()

# Frop Out- a regularization technique that helps prevent overfitting in neural networks. During training, dropout randomly sets a fraction of the neurons to zero, preventing the model from relying too heavily on any single neuron. Dropout is applied during training by randomly zeroing out a fraction of the input elements, controlled by the dropout rate p. The output is represented as
dropout_layer = nn.Dropout(p=0.5)
dropout_layer = tf.keras.layers.Dropout(rate=0.5)

# LSTM - address the vanishing gradient problem in traditional RNNs, enabling them to learn long-range dependencies in sequences. LSTM layers use a gating mechanism with three gates (input, forget, and output) and an internal cell state to control the flow of information through the network. 
lstm_layer = nn.LSTM(input_size=64, hidden_size=64)
lstm_layer = tf.keras.layers.LSTM(units=64)


# Attention Layers 
attention_layer = tf.keras.layers.Attention()
# PyTorch has no simple attention layer implementation, so it can be defined via custom class Attention or also MultiHeadAttention can be used
import torch
import torch.nn.functional as F
import torch.nn as nn

class Attention(torch.nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)

    def forward(self, input):
        Q = self.query(input)
        K = self.key(input)
        V = self.value(input)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        weights = F.softmax(scores, dim=-1)

        return torch.matmul(weights, V)

#or with multihead:
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=1)


'''
tf.keras provides functionalities to save and load models. This allows for preserving the model's architecture, weights, and training configuration, enabling you to:
Resume training: Continue training from a saved point without starting from scratch.
Share models: Distribute trained models for others to use or fine-tune.
Deploy models: Export models for use in production environments without needing the original Python code.

'''
import tensorflow as tf

# Create a simple Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model (optional, but recommended for saving optimizer state)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Save the model in the default Keras v3 format
model.save('my_model.keras')

# Or save in TensorFlow SavedModel format
# model.save('my_saved_model', save_format='tf')

# Or save in HDF5 format
# model.save('my_model.h5', save_format='h5')

# Load the saved model
loaded_model = tf.keras.models.load_model('my_model.keras')

# Now you can use the loaded_model for predictions or further training
predictions = loaded_model.predict(some_data)


import torch
from torch.export import export, ExportedProgram

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: ExportedProgram = export(Mod(), args=example_args)
print(exported_program)

# torch.export-based ONNX Exporter
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)

