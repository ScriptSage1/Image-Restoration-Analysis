import torch
import torch.nn as nn
import numpy as np

# This is the final, correct blueprint for the 4-channel color denoising model.

def sequential(*args):
    """A sequential container for modules."""
    if len(args) == 1:
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ResBlock(nn.Module):
    """Residual block."""
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.res = sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        res = self.res(x)
        return x + res

class DRUNet(nn.Module):
    """DRUNet model for image denoising (4-channel input)."""
    def __init__(self, in_channels=4, out_channels=3, n_feat=64, n_block=4):
        super(DRUNet, self).__init__()
        self.m_head = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=False)

        # Downsampling with 2x2 kernels
        self.m_down1 = sequential(*[ResBlock(n_feat, n_feat) for _ in range(n_block)], nn.Conv2d(n_feat, n_feat*2, 2, 2, bias=False))
        self.m_down2 = sequential(*[ResBlock(n_feat*2, n_feat*2) for _ in range(n_block)], nn.Conv2d(n_feat*2, n_feat*4, 2, 2, bias=False))
        self.m_down3 = sequential(*[ResBlock(n_feat*4, n_feat*4) for _ in range(n_block)], nn.Conv2d(n_feat*4, n_feat*8, 2, 2, bias=False))

        self.m_body = sequential(*[ResBlock(n_feat*8, n_feat*8) for _ in range(n_block)])

        # Upsampling
        self.m_up3 = sequential(nn.ConvTranspose2d(n_feat*8, n_feat*4, 2, 2, bias=False), *[ResBlock(n_feat*4, n_feat*4) for _ in range(n_block)])
        self.m_up2 = sequential(nn.ConvTranspose2d(n_feat*4, n_feat*2, 2, 2, bias=False), *[ResBlock(n_feat*2, n_feat*2) for _ in range(n_block)])
        self.m_up1 = sequential(nn.ConvTranspose2d(n_feat*2, n_feat, 2, 2, bias=False), *[ResBlock(n_feat, n_feat) for _ in range(n_block)])

        self.m_tail = nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        h, w = x.size()[-2:]
        # Ensure padding calculation is correct and works for all input sizes
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        x = nn.ReplicationPad2d((0, pad_w, 0, pad_h))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        # Crop back to original size
        return x[..., :h, :w]

