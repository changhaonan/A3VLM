import warnings
import torch
import torch.nn as nn

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")

    class RMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            """
            Initialize the RMSNorm normalization layer.

            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.

            """
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            """
            Apply the RMSNorm normalization to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.

            """
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            """
            Forward pass through the RMSNorm layer.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after applying RMSNorm.

            """
            output = self._norm(x.float()).type_as(x)
            return output * self.weight




#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024 Baifeng Shi.
#  All rights reserved.
#
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch

def split_chessboard(x, num_split):
    """
        x: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
    """
    B, C, H, W = x.shape
    assert H % num_split == 0 and W % num_split == 0
    h, w = H // num_split, W // num_split
    x_split = torch.cat([x[:, :, i*h:(i+1)*h, j*w:(j+1)*w] for i in range(num_split) for j in range(num_split)], dim=0)
    return x_split

def merge_chessboard(x, num_split):
    """
        x: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        (inverse of split_chessboard)
    """
    B, C, H, W = x.shape
    assert B % (num_split**2) == 0
    b = B // (num_split**2)
    x_merge = torch.cat([torch.cat([x[(i*num_split + j)*b:(i*num_split + j + 1)*b] for j in range(num_split)], dim=-1)
                         for i in range(num_split)], dim=-2)
    return x_merge