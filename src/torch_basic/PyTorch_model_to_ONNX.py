

import torch
print(torch.__version__)

import onnxscript
print(onnxscript.__version__)

import onnxruntime
print(onnxruntime.__version__)


import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Open Neural Network eXchange (ONNX) is an open standard format for representing machine learning models. The torch.onnx module provides APIs to capture the computation graph from a native PyTorch torch.nn.Module model and convert it into an ONNX graph.
The exported model can be consumed by any of the many runtimes that support ONNX, including Microsoft's ONNX Runtime.

ONNX Runtime provides an efficient, scalable, and flexible solution for deploying machine learning models in production environments across a variety of devices.
ONNX is an open standard for representing machine learning models, allowing them to be transferred between different frameworks,  such as PyTorch, TensorFlow, and Scikit-learn. The model can be deployed in production environment across a various of devices, like Widows, Linux and macOS.

"""
# exported our PyTorch model to ONNX format, saved the model to disk, viewed it using Netron, executed it with ONNX Runtime and finally compared its numerical results with PyTorch’s.

### Create a simple image classifier model
class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

torch_model = ImageClassifierModel()
print(torch_model)


#### Export the model to ONNX format
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 1, 32, 32),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)


### optional: Optimize the ONNX model
onnx_program.optimize()
# We do not need any code change to the model. The resulting ONNX model is stored within torch.onnx.ONNXProgram as a binary protobuf file.


### Save the ONNX model in a file
onnx_program.save("image_classifier_model.onnx")
# load the ONNX file back into memory 
import onnx

onnx_model = onnx.load("image_classifier_model.onnx")
onnx.checker.check_model(onnx_model)


### Visualize the ONNX model graph using Netron. 
# Netron can either be installed on macos, Linux or Windows computers, or run directly from the browser.  The web version link: https://netron.app/


### Execute the ONNX model with ONNX Runtime
import onnxruntime

onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
# print(f"Input length: {len(onnx_inputs)}")
# print(f"Sample input: {onnx_inputs}")

ort_session = onnxruntime.InferenceSession(
    "./image_classifier_model.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]



# Compare the PyTorch results with the ones from the ONNX Runtime
torch_outputs = torch_model(*example_inputs)

assert len(torch_outputs) == len(onnxruntime_outputs)
for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
    torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

print("PyTorch and ONNX Runtime output matched!")
print(f"Output length: {len(onnxruntime_outputs)}")
print(f"Sample output: {onnxruntime_outputs}")