import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np

from makeFace.layer import NoiseLayer, PixelNormLayer, StyleMod, MyLinear, MyConv2d, BlurLayer


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(
        self,
        channels,
        dlatent_size,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(("noise", NoiseLayer(channels)))
        layers.append(("activation", activation_layer))
        if use_pixel_norm:
            layers.append(("pixel_norm", PixelNormLayer()))
        if use_instance_norm:
            layers.append(("instance_norm", nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):
    def __init__(
        self,
        nf,
        dlatent_size,
        const_input_layer,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = MyLinear(
                dlatent_size, nf * 16, gain=gain / 4, use_wscale=use_wscale
            )  # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(
            nf,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class GSynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        blur_filter,
        dlatent_size,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            gain=gain,
            use_wscale=use_wscale,
            intermediate=blur,
            upscale=True,
        )
        self.epi1 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv1 = MyConv2d(
            out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale
        )
        self.epi2 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x
