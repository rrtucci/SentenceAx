"""

This file illustrates the use of torch.nn.CrossEntropyLoss

Ref:
https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

"""
import torch
import torch.nn as nn

# Sample input tensor with shape (5, 4)
input_tensor = torch.randn(5, 4)

# Sample target tensor with shape (5, ) containing class labels
target_tensor = torch.tensor([2, 0, 1, 3, -100])

# Create the CrossEntropy loss criterion
loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

# Calculate the loss
loss = loss_fun(input_tensor, target_tensor)

#input_tensor.shape = (N, N_C)
#target_tensor.shape = (N,).
#target_tensor[i] \in {0, 1, ..., N_C-1} so non-negative

print(loss.item())