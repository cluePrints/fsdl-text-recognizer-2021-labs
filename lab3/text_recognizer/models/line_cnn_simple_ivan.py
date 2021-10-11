import torch
import torch.nn as nn

from typing import Dict, Any

from argparse import _ArgumentGroup

import argparse
import numpy as np

from .cnn import CNN, IMAGE_SIZE

WINDOW_WIDTH = 28
WINDOW_STRIDE = 28

class LineCNNSimpleIvan(nn.Module):
    """Process the line through a CNN and process the resulting sequence through LSTM layers."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.window_width = args.window_width
        self.window_stride = args.window_stride
        self.data_config = data_config
        self.cnn = CNN(data_config=data_config, args=args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Batch, Color, Height, Width]
        """
        DIM_H=2
        DIM_W=3
        n_images, n_colors, height, width = x.shape
        w_starts = np.arange(0, width, self.window_stride)
        w_ends = w_starts + self.window_width
        res = []
        for w_start, w_end in zip(w_starts, w_ends):
            im = x[:,:,:, w_start:w_end]
            # flip to (B, C, H, W) for CNN
            im = torch.transpose(im, DIM_H, DIM_W)
            # (B, C) returned, make it (B, C, 1) to enable concat
            r = self.cnn(im).unsqueeze(2)
            res.append(r)

        return torch.cat(res, dim=2)

    def add_to_argparse(parser: _ArgumentGroup):
        parser.add_argument('--window_width', type=int)
        parser.add_argument('--window_stride', type=int)
