from typing import Any, Dict, Union, Tuple
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Common type hints
Param2D = Union[int, Tuple[int, int]]

CONV_DIM = 32
FC_DIM = 512
WINDOW_WIDTH = 16
WINDOW_STRIDE = 8


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Param2D = 3,
        stride: Param2D = 1,
        padding: Param2D = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


class LineCNN(nn.Module):
    """
    Model that uses a simple CNN to process an image of a line of characters with a window, outputs a sequence of logits
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]

        _C, H, _W = data_config["input_dims"]
        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        """
        In: torch.Size([128, 1, 28, 896]).  Conv: Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        In: torch.Size([128, 32, 28, 896]). Conv: Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        In: torch.Size([128, 32, 28, 896]). Conv: Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        In: torch.Size([128, 32, 14, 448]). Conv: Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        In: torch.Size([128, 32, 14, 448]). Conv: Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        In: torch.Size([128, 64, 7, 224]).  Conv: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        In: torch.Size([128, 64, 7, 224]).  Conv: Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        In: torch.Size([128, 128, 4, 112]). Conv: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        In: torch.Size([128, 128, 4, 112]). Conv: Conv2d(128, 512, kernel_size=(3, 3), stride=(3, 3))
        Out: torch.Size([128, 512, 1, 37])

        stride 2 with padding 1 means we're halving the dims every time
        squeeze following this in the next layer sort of relies on H being 1
        last block basically plays the same sliding window role the app code did in LineCNNSimple
        we're adjusting the window size (which is kernel size now) and stride (which is kernel stride) to account for dim halving we've done previously
        that sort of keeps the receptive field somewhat similar to the manual sliding one
        """
        # Input is (1, H, W)
        self.convs = nn.Sequential(
            ConvBlock(1, conv_dim),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim, stride=2),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim * 2, stride=2),
            ConvBlock(conv_dim * 2, conv_dim * 2),
            ConvBlock(conv_dim * 2, conv_dim * 4, stride=2),
            ConvBlock(conv_dim * 4, conv_dim * 4),
            ConvBlock(
                conv_dim * 4, fc_dim, kernel_size=(H // 8, self.WW // 8), stride=(H // 8, self.WS // 8), padding=0
            ),
        )
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights in a better way than default.
        See https://github.com/pytorch/pytorch/issues/18182
        """
        for m in self.modules():
            if type(m) in {
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.Linear,
            }:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(  # pylint: disable=protected-access
                        m.weight.data
                    )
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            (B, 1, H, W) input image

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and self.window_width
            C is self.num_classes
        """
        _B, _C, _H, _W = x.shape
        # for layer in self.convs:
        #     print(f"In: {x.shape}. Conv: {layer.conv}")
        #     x = layer(x)

        # print(f"Out: {x.shape}")
        x = self.convs(x)  # (B, FC_DIM, 1, Sx)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, S, FC_DIM)
        x = F.relu(self.fc1(x))  # -> (B, S, FC_DIM)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, S, C)
        x = x.permute(0, 2, 1)  # -> (B, C, S)
        if self.limit_output_length:
            x = x[:, :, : self.output_length]
        print(f"{x.shape}")
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument(
            "--window_width",
            type=int,
            default=WINDOW_WIDTH,
            help="Width of the window that will slide over the input image.",
        )
        parser.add_argument(
            "--window_stride",
            type=int,
            default=WINDOW_STRIDE,
            help="Stride of the window that will slide over the input image.",
        )
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser
