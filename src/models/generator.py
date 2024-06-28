import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self,
            latent_size: int,
            hidden_1: int,
            hidden_2: int,
            hidden_3: int,
            hidden_4: int
        ) -> None:
        super().__init__()

        # From (`batch_szie`, `latent_size`, 1, 1) to (`batch_szie`, `hidden_1`, 4, 4)
        self.deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size,
                               out_channels=hidden_1,
                               kernel_size=(4, 4),
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(hidden_1),
            nn.ReLU(inplace=True)
        )

        # Output: (`batch_size`, `hidden_2`, 8, 8)
        self.deconv_block_2 = Generator._deconv_block(hidden_1, hidden_2)

        # Output: (`batch_size`, `hidden_3`, 16, 16)
        self.deconv_block_3 = Generator._deconv_block(hidden_2, hidden_3)

        # Output: (`batch_size`, `hidden_4`, 32, 32)
        self.deconv_block_4 = Generator._deconv_block(hidden_3, hidden_4)

        # Output: (`batch_size`, 3, 64, 64)
        self.deconv_block_5 = nn.ConvTranspose2d(
            in_channels=hidden_4,
            out_channels=3,
            kernel_size=(4, 4),
            stride=2,
            padding=1,
            bias=False
        )

        # Non-Linear Activation Function
        self.non_linearity = nn.Tanh()

    @staticmethod
    def _deconv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.non_linearity(self.deconv_block_5(self.deconv_block_4(self.deconv_block_3(self.deconv_block_2(self.deconv_block_1(x))))))
