import torch
import torch.nn as nn
import numpy as np


def CircularPad2d(input, pad_size):
    output = torch.cat((input[:,:,-pad_size:, :], input, input[:, :, :pad_size, :]), 2)
    output = torch.cat((output[:, :, :, -pad_size:], output, output[:, :, :, :pad_size]), 3)
    return output


class TransformerNet(torch.nn.Module):
    def __init__(self, noise_dim=2):
        super(TransformerNet, self).__init__()

        conv_num = 256
        conv_num1 = int(conv_num / 2)
        # Upsampling layers
        # self.deconv1 = UpsampleConvLayer(noise_dim, conv_num, kernel_size = 8, stride = 1)


        # 8 * 8

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.in2 = InstanceNormalization(1)

        self.conv1 = ConvLayer(1, 32, kernel_size=3, stride=1)
        self.in3 = InstanceNormalization(32)

        self.res1 = ResidualBlock(32)

        # 16 * 16
        # ----------------------------------------------------------

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.in6 = InstanceNormalization(32)

        self.conv4 = ConvLayer(32, 128, kernel_size=3, stride=1)
        self.in7 = InstanceNormalization(128)

        self.res2 = ResidualBlock(128)

        # 32 * 32
        # -----------------------------------------------------------------

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.in10 = InstanceNormalization(128)

        self.conv7 = ConvLayer(128, 32, kernel_size=3, stride=1)
        self.in11 = InstanceNormalization(32)

        self.res3 = ResidualBlock(32)

        # 64 * 64
        # -----------------------------------------------------------------
        self.conv8 = ConvLayer(32, 16, kernel_size=3, stride=1)
        self.conv9 = ConvLayer(16, 8, kernel_size=3, stride=1)
        self.conv10 = ConvLayer(8, 1, kernel_size=3, stride=1)


        # Non-linearities
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, X):

        in_X = X

        # ----------------------------------------------------------

        y = self.in2(self.up1(in_X))
        # y = self.up1(in_X)
        y = self.relu(self.in3(self.conv1(y)))
        y = self.res1(y)
        # y = self.relu(self.in4(self.conv2(y)))
        # y = self.relu(self.in5(self.conv3(y)))
        # ----------------------------------------------------------

        y = self.in6(self.up2(y))
        y = self.relu(self.in7(self.conv4(y)))
        y = self.res2(y)
        # y = self.relu(self.in8(self.conv5(y)))
        # y = self.relu(self.in9(self.conv6(y)))

        # 32 * 32
        # ----------------------------------------------------------

        y = self.in10(self.up3(y))
        y = self.relu(self.in11(self.conv7(y)))
        y = self.res3(y)
        # y = self.relu(self.in12(self.conv8(y)))
        # y = self.relu(self.in13(self.conv9(y)))

        # 64 * 64
        # ----------------------------------------------------------
        y = self.conv8(y)
        y = self.conv9(y)
        y = self.conv10(y)

        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        # reflection_padding = int(np.floor(kernel_size / 2))
        # self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        # zero_padding = int(np.floor(kernel_size / 2))
        # self.zero_pad = nn.ZeroPad2d(zero_padding)

        # replicate_padding = int(np.floor(kernel_size / 2))
        # self.replicate_pad = nn.ReplicationPad2d(replicate_padding)

        self.circular_padding = int(np.floor(kernel_size / 2))

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # out = self.zero_pad(x)
        # out = self.replicate_pad(x)
        # out = self.reflection_pad(x)
        out = CircularPad2d(x, self.circular_padding)
        out = self.conv2d(out)

        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out