import torch
import torch.nn as nn

from typing import Dict, Any

from argparse import _ArgumentGroup

import argparse
import numpy as np

from .cnn import CNN, IMAGE_SIZE


class LineCNNIvan(nn.Module):
    """Process the line through a CNN and process the resulting sequence through LSTM layers."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        conv_dim = args.fc_dim
        n_classes = len(data_config['mapping'])
        n_colors, HEIGHT, _W = data_config["input_dims"]
        seq_len, _ = data_config['output_dims']
        fc_dim = args.fc_dim
        self.window_width = args.window_width
        self.window_height = args.window_width
        self.window_stride = args.window_stride
        self.data_config = data_config
        self.cnn = CNN(data_config=data_config, args=args)
        # striding early (vs in the last layer) allows us to reduce the amount of compute while still passing info down the line
        self.conv1 = nn.Conv2d(n_colors, conv_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(conv_dim, fc_dim,
                                kernel_size=(HEIGHT//8, self.window_width//8), stride=(self.window_height//8, self.window_stride//8), padding=0
        )
        # TODO: this sort of gets to 27x27 receptive field on a single output of the feature map
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_dim, n_classes)
        self.limit_output_length = args.limit_output_length
        self.output_length = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Batch, Color, Height, Width] -> [Batch, Classes, SeqLen]
        1) receptive field size for each of the output columns ~window_width
        2) there-s gonna be (line_width-window_width)/window_stride columns in the output
        last? convolutional layer will be basically doing the job the code was doing of sliding the CNN
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # we're at (B, fc_dim, 1, SeqLen) here
        # -> (B, SeqLen, n_classes)
        x = x.squeeze(2).transpose(1, 2)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # -> (B, n_classes, SeqLen)
        x = x.transpose(1, 2)

        # -> torch.Size([128, 83, 32])
        if self.limit_output_length:
            x = x[..., :self.output_length]

        return x

    def add_to_argparse(parser: _ArgumentGroup):
        parser.add_argument('--window_width', type=int)
        parser.add_argument('--window_stride', type=int)
        parser.add_argument('--fc_dim', type=int, default=128)
        parser.add_argument('--conv_dim', type=int, default=128)
        parser.add_argument("--limit_output_length", action="store_true", default=False)
