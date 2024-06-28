import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self,
            hidden_1: int,
            hidden_2: int,
            hidden_3: int,
            hidden_4: int
        ) -> None:
        super().__init__()

        # Output: (`batch_size`, `hidden_1`, 32, 32)
        self.conv_block_1 = Discriminator._conv_block(3, hidden_1)

        # Output: (`batch_size`, `hidden_2`, 16, 16)
        self.conv_block_2 = Discriminator._conv_block(hidden_1, hidden_2)

        # Output: (`batch_size`, `hidden_3`, 8, 8)
        self.conv_block_3 = Discriminator._conv_block(hidden_2, hidden_3)

        # Output: (`batch_size`, `hidden_4`, 4, 4)
        self.conv_block_4 = Discriminator._conv_block(hidden_3, hidden_4)

        # Output: (`batch_size`, 1, 1, 1)
        self.conv_block_5 = nn.Conv2d(
            in_channels=hidden_4,
            out_channels=1,
            kernel_size=(4, 4),
            stride=1,
            padding=0,
            bias=False
        )

        # Output: (`batch_size`, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Sigmoid()
        )

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_5(self.conv_block_4(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))))
