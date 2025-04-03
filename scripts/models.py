# models.py
import torch
import torch.nn as nn
from layers import DoubleConvMaSA
import logging

logger = logging.getLogger(__name__)

class AutoencoderMaSA(nn.Module):
    def __init__(self, height, width, channels, savename="autoencoder_masa", use_masa=True, gamma=0.9):
        super(AutoencoderMaSA, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.savename = savename
        filters = 32  # Keep reduced filters
        dropout_val = 0

        # Encoder
        self.conv_224 = DoubleConvMaSA(channels, filters)
        self.conv_224_att = DoubleConvMaSA(channels, filters, use_masa=use_masa, gamma=gamma)
        self.pool_112 = nn.MaxPool2d(2)
        self.conv_112 = DoubleConvMaSA(filters, 2 * filters)
        self.conv_112_att = DoubleConvMaSA(filters, 2 * filters, use_masa=use_masa, gamma=gamma)
        self.pool_56 = nn.MaxPool2d(2)
        self.conv_56 = DoubleConvMaSA(2 * filters, 4 * filters)
        self.conv_56_att = DoubleConvMaSA(2 * filters, 4 * filters, use_masa=use_masa, gamma=gamma)
        self.pool_28 = nn.MaxPool2d(2)
        self.conv_28 = DoubleConvMaSA(4 * filters, 8 * filters)
        self.pool_14 = nn.MaxPool2d(2)
        self.conv_14 = DoubleConvMaSA(8 * filters, 16 * filters)
        self.pool_7 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16 * filters, 32 * filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * filters),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up_14 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_14 = DoubleConvMaSA(32 * filters + 16 * filters, 16 * filters)
        self.up_28 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_28 = DoubleConvMaSA(16 * filters + 8 * filters, 8 * filters)
        self.up_56 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_56 = DoubleConvMaSA(8 * filters + 4 * filters, 4 * filters)
        self.up_112 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_112 = DoubleConvMaSA(4 * filters + 2 * filters, 2 * filters)
        self.up_224 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_224 = DoubleConvMaSA(2 * filters + filters, filters, dropout=dropout_val)

        # Final layer
        self.conv_final = nn.Conv2d(filters, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        conv_224 = self.conv_224(x)
        conv_224_att = self.conv_224_att(x)
        pool_112 = self.pool_112(conv_224)
        conv_112 = self.conv_112(pool_112)
        conv_112_att = self.conv_112_att(pool_112)
        pool_56 = self.pool_56(conv_112)
        conv_56 = self.conv_56(pool_56)
        conv_56_att = self.conv_56_att(pool_56)
        pool_28 = self.pool_28(conv_56)
        conv_28 = self.conv_28(pool_28)
        pool_14 = self.pool_14(conv_28)
        conv_14 = self.conv_14(pool_14)
        pool_7 = self.pool_7(conv_14)

        # Bottleneck
        conv_7 = self.bottleneck(pool_7)

        # Decoder with skip connections
        up_14 = torch.cat([self.up_14(conv_7), conv_14], dim=1)
        up_conv_14 = self.up_conv_14(up_14)
        up_28 = torch.cat([self.up_28(up_conv_14), conv_28], dim=1)
        up_conv_28 = self.up_conv_28(up_28)
        up_56 = torch.cat([self.up_56(up_conv_28), conv_56_att], dim=1)
        up_conv_56 = self.up_conv_56(up_56)
        up_112 = torch.cat([self.up_112(up_conv_56), conv_112_att], dim=1)
        up_conv_112 = self.up_conv_112(up_112)
        up_224 = torch.cat([self.up_224(up_conv_112), conv_224_att], dim=1)
        up_conv_224 = self.up_conv_224(up_224)

        # Reconstruction
        conv_final = self.conv_final(up_conv_224)
        output = self.sigmoid(conv_final)
        return output, conv_7

    def get_embedding(self, x):
        conv_224 = self.conv_224(x)
        pool_112 = self.pool_112(conv_224)
        conv_112 = self.conv_112(pool_112)
        pool_56 = self.pool_56(conv_112)
        conv_56 = self.conv_56(pool_56)
        pool_28 = self.pool_28(conv_56)
        conv_28 = self.conv_28(pool_28)
        pool_14 = self.pool_14(conv_28)
        conv_14 = self.conv_14(pool_14)
        pool_7 = self.pool_7(conv_14)
        embedding = self.bottleneck(pool_7)
        return embedding



class MaSAUNetSegmentation(nn.Module):
    def __init__(self, height, width, in_channels, out_channels, autoencoder=None, use_masa=True, gamma=0.9):
        super(MaSAUNetSegmentation, self).__init__()
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.autoencoder = autoencoder  # Pre-trained AutoencoderMaSA instance
        filters = 32  # Starting filter size, consistent with autoencoder
        dropout_val = 0

        # Encoder
        self.conv_224 = DoubleConvMaSA(in_channels, filters)
        self.conv_224_att = DoubleConvMaSA(in_channels, filters, use_masa=use_masa, gamma=gamma)
        self.pool_112 = nn.MaxPool2d(2)
        self.conv_112 = DoubleConvMaSA(filters, 2 * filters)
        self.conv_112_att = DoubleConvMaSA(filters, 2 * filters, use_masa=use_masa, gamma=gamma)
        self.pool_56 = nn.MaxPool2d(2)
        self.conv_56 = DoubleConvMaSA(2 * filters, 4 * filters)
        self.conv_56_att = DoubleConvMaSA(2 * filters, 4 * filters, use_masa=use_masa, gamma=gamma)
        self.pool_28 = nn.MaxPool2d(2)
        self.conv_28 = DoubleConvMaSA(4 * filters, 8 * filters)
        self.pool_14 = nn.MaxPool2d(2)
        self.conv_14 = DoubleConvMaSA(8 * filters, 16 * filters)
        self.pool_7 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16 * filters, 32 * filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 * filters),
            nn.ReLU(inplace=True)
        )
        # Adjustment layer to combine autoencoder bottleneck with segmentation bottleneck
        if autoencoder is not None:
            self.bottleneck_adjust = nn.Conv2d(32 * filters * 2, 32 * filters, kernel_size=1)

        # Decoder
        self.up_14 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_14 = DoubleConvMaSA(32 * filters + 16 * filters, 16 * filters)
        self.up_28 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_28 = DoubleConvMaSA(16 * filters + 8 * filters, 8 * filters)
        self.up_56 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_56 = DoubleConvMaSA(8 * filters + 4 * filters, 4 * filters)
        self.up_112 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_112 = DoubleConvMaSA(4 * filters + 2 * filters, 2 * filters)
        self.up_224 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv_224 = DoubleConvMaSA(2 * filters + filters, filters, dropout=dropout_val)

        # Final layer for segmentation
        self.conv_final = nn.Conv2d(filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        conv_224 = self.conv_224(x)
        conv_224_att = self.conv_224_att(x)
        pool_112 = self.pool_112(conv_224)
        conv_112 = self.conv_112(pool_112)
        conv_112_att = self.conv_112_att(pool_112)
        pool_56 = self.pool_56(conv_112)
        conv_56 = self.conv_56(pool_56)
        conv_56_att = self.conv_56_att(pool_56)
        pool_28 = self.pool_28(conv_56)
        conv_28 = self.conv_28(pool_28)
        pool_14 = self.pool_14(conv_28)
        conv_14 = self.conv_14(pool_14)
        pool_7 = self.pool_7(conv_14)

        # Bottleneck
        conv_7 = self.bottleneck(pool_7)
        if self.autoencoder is not None:
            # Get the bottleneck from the pre-trained autoencoder
            with torch.no_grad():  # No gradients for autoencoder during segmentation
                _, autoencoder_bottleneck = self.autoencoder(x)
            # Concatenate and adjust channels
            conv_7 = torch.cat([conv_7, autoencoder_bottleneck], dim=1)
            conv_7 = self.bottleneck_adjust(conv_7)

        # Decoder with skip connections
        up_14 = torch.cat([self.up_14(conv_7), conv_14], dim=1)
        up_conv_14 = self.up_conv_14(up_14)
        up_28 = torch.cat([self.up_28(up_conv_14), conv_28], dim=1)
        up_conv_28 = self.up_conv_28(up_28)
        up_56 = torch.cat([self.up_56(up_conv_28), conv_56_att], dim=1)
        up_conv_56 = self.up_conv_56(up_56)
        up_112 = torch.cat([self.up_112(up_conv_56), conv_112_att], dim=1)
        up_conv_112 = self.up_conv_112(up_112)
        up_224 = torch.cat([self.up_224(up_conv_112), conv_224_att], dim=1)
        up_conv_224 = self.up_conv_224(up_224)

        # Segmentation output
        conv_final = self.conv_final(up_conv_224)
        output = self.sigmoid(conv_final)
        return output
