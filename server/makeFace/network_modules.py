import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np

from makeFace.layer import PixelNormLayer, MyLinear, MyConv2d
from makeFace.network_utils import InputBlock, GSynthesisBlock


# Generator Mapping Network
class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity="lrelu", use_wscale=True):
        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[nonlinearity]
        layers = [
            ("pixel_norm", PixelNormLayer()),
            (
                "dense0",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense0_act", act),
            (
                "dense1",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense1_act", act),
            (
                "dense2",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense2_act", act),
            (
                "dense3",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense3_act", act),
            (
                "dense4",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense4_act", act),
            (
                "dense5",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense5_act", act),
            (
                "dense6",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense6_act", act),
            (
                "dense7",
                MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),
            ),
            ("dense7_act", act),
        ]
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.register_buffer("avg_latent", avg_latent)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return torch.where(do_trunc, interp, x)


# Generator Synthesis Network
class G_synthesis(nn.Module):
    def __init__(
        self,
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        num_channels=3,  # Number of output color channels.
        resolution=1024,  # Output resolution.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        use_styles=True,  # Enable style inputs?
        const_input_layer=True,  # First layer is a learned constant?
        use_noise=True,  # Enable noise inputs?
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        nonlinearity="lrelu",  # Activation function: 'relu', 'lrelu'
        use_wscale=True,  # Enable equalized learning rate?
        use_pixel_norm=False,  # Enable pixelwise feature vector normalization?
        use_instance_norm=True,  # Enable instance normalization?
        dtype=torch.float32,  # Data type to use for activations and outputs.
        blur_filter=[
            1,
            2,
            1,
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
    ):
        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))

        assert resolution == 2**resolution_log2 and resolution >= 4

        act, gain = {
            "relu": (torch.relu, np.sqrt(2)),
            "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),
        }[nonlinearity]

        num_layers = resolution_log2 * 2 - 2
        num_styles = num_layers if use_styles else 1

        torgbs = []
        blocks = []

        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            name = "{s}x{s}".format(s=2**res)
            if res == 2:
                blocks.append(
                    (
                        name,
                        InputBlock(
                            channels,
                            dlatent_size,
                            const_input_layer,
                            gain,
                            use_wscale,
                            use_noise,
                            use_pixel_norm,
                            use_instance_norm,
                            use_styles,
                            act,
                        ),
                    )
                )

            else:
                blocks.append(
                    (
                        name,
                        GSynthesisBlock(
                            last_channels,
                            channels,
                            blur_filter,
                            dlatent_size,
                            gain,
                            use_wscale,
                            use_noise,
                            use_pixel_norm,
                            use_instance_norm,
                            use_styles,
                            act,
                        ),
                    )
                )
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, dlatents_in):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        batch_size = dlatents_in.size(0)
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2 * i : 2 * i + 2])
            else:
                x = m(x, dlatents_in[:, 2 * i : 2 * i + 2])
        rgb = self.torgb(x)
        return rgb
