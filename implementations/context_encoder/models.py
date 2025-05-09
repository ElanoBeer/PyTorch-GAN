import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        # Input: 128x128 → Output: 128x128
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),  # 128 → 64
            *downsample(64, 128),                        # 64 → 32
            *downsample(128, 256),                       # 32 → 16
            *downsample(256, 512),                       # 16 → 8

            *upsample(512, 256),                         # 8 → 16
            *upsample(256, 128),                         # 16 → 32
            *upsample(128, 64),                          # 32 → 64
            *upsample(64, 32),                           # 64 → 128

            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Create a PatchGAN discriminator appropriate for 128×128 patches
        # (The patch size will be determined in context_encoder.py)
        layers = []
        in_filters = channels

        # Downsample from 128x128 to 8x8 for PatchGAN
        for out_filters, stride, normalize in [
            (64, 2, False),  # 128 → 64
            (128, 2, True),  # 64 → 32
            (256, 2, True),  # 32 → 16
            (512, 2, True),  # 16 → 8
        ]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output a single value per patch
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)