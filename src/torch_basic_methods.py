""""

"""

import numpy as np
import torch

'''
torch.squeeze(input: Tensor, dim: Optional[Union[int, List[int]]]) → Tensor
Remove dimension of size 1 from a tensor.
For example, remove batch dimension when it is become unnecessary, clean up after some operations like convolutions with specific padding or reshaping.

If input is of shape: (A×1×B×C×1×D) then the input.squeeze() will be of shape:(A×B×C×D).
When dim is given, a squeeze operation is done only in the given dimension(s). If input is of shape: (A×1×B), squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A×B).

torch.unsqueeze(input, dim) → Tensor
Adds a new dimension of size one to a tensor at a specified position. 
A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used.
'''
x = torch.zeros(2, 1, 2, 1, 2)
print(x)
# the following does nothing
print(torch.squeeze(x,0))
print(torch.squeeze(x,0).size())

# this remove the the 2nd domension only not the 4th
print(torch.squeeze(x,1))
print(torch.squeeze(x,1).size())

# the following does nothing
print(torch.squeeze(x,2))
print(torch.squeeze(x,2).size())


x = torch.tensor([[1, 2, 3, 4],[5,6,7,8]]) 
x.dim() # 2
x.size() # torch.Size([2, 4])
# decide the dim argument based on the returned size. for example, if dim=0 | -3, a new dimension of size one will be added before the first or the thrid from the last.
torch.unsqueeze(x, 3).size()  # IndexError: Dimension out of range

torch.unsqueeze(x, 2).size()  # torch.Size([2, 4, 1])
torch.unsqueeze(x, -1).size() # torch.Size([2, 4, 1])

torch.unsqueeze(x, 1).size()  # torch.Size([2, 1, 4])
torch.unsqueeze(x, -2).size() # torch.Size([2, 1, 4])

torch.unsqueeze(x, 0).size()  # torch.Size([1, 2, 4])
torch.unsqueeze(x, -3).size() # torch.Size([1, 2, 4])

torch.unsqueeze(x, -4).size() # IndexError: Dimension out of range

