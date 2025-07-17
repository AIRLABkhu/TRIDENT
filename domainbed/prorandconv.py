#!/usr/bin/env python
"""
convolution layer whose weights can be randomized

Created by zhenlinxu on 12/28/2019
"""
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
import random
import collections

class RandConvModule(nn.Module):
    def __init__(self, net=None, kernel_size=3, in_channels=3, out_channels=3,
                 rand_bias=False,
                 mixing=False,
                 identity_prob=0.0, distribution='kaiming_normal',
                 data_mean=None, data_std=None, clamp_output=False,
                 ):
        """

        :param net:
        :param kernel_size:
        :param in_channels:
        :param out_channels:
        :param rand_bias:
        :param mixing: "random": output = (1-alpha)*input + alpha* randconv(input) where alpha is a random number sampled
                            from a distribution defined by res_dist
        :param identity_prob:
        :param distribution:
        :param data_mean:
        :param data_std:
        :param clamp_output:
        """

        super(RandConvModule, self).__init__()

        # if the input is not normalized, we need to normalized with given mean and std (tensor of size 3)
        self.register_buffer('data_mean', None if data_mean is None else torch.tensor(data_mean).reshape(3, 1, 1))
        self.register_buffer('data_std', None if data_std is None else torch.tensor(data_std).reshape(3, 1, 1))

        # adjust output range based on given data mean and std, (clamp or norm)
        # clamp with clamp the value given that the was image pixel values [0,1]
        # normalize will linearly rescale the values to the allowed range
        # The allowed range is ([0, 1]-data_mean)/data_std in each color channel
        self.clamp_output = clamp_output
        if self.clamp_output:
            assert (self.data_mean is not None) and (self.data_std is not None), "Need data mean/std to do output range adjust"
        self.register_buffer('range_up', None if not self.clamp_output else (torch.ones(3).reshape(3, 1, 1) - self.data_mean) / self.data_std)
        self.register_buffer('range_low', None if not self.clamp_output else (torch.zeros(3).reshape(3, 1, 1) - self.data_mean) / self.data_std)

        if isinstance(kernel_size, collections.Sequence) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        if mixing:
            out_channels = in_channels

        # generate random conv layer
        print("Add RandConv layer with kernel size {}, output channel {}".format(kernel_size, out_channels))
        self.randconv = MultiScaleRandConv2d(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_size,
                                             stride=1, rand_bias=rand_bias,
                                             distribution=distribution,
                                             clamp_output=self.clamp_output,
                                             range_low=self.range_low,
                                             range_up=self.range_up,
                                             )


        # mixing mode
        self.mixing = mixing # In the mixing mode, a mixing connection exists between input and output of random conv layer
        # self.res_dist = res_dist
        self.res_test_weight = None
        if self.mixing:
            assert in_channels == out_channels or out_channels == 1, \
                'In mixing mode, in/out channels have to be equal or out channels is 1'
            self.alpha = random.random()  # sample mixing weights from uniform distributin (0, 1)

        self.identity_prob = identity_prob  # the probability that use original input

    def forward(self, input):
        """assume that the input is whightened"""

        ######## random conv ##########
        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            # whiten input and go through randconv
            output = self.randconv(input)

            if self.mixing:
                output = (self.alpha*output + (1-self.alpha)*input)

            if self.clamp_output:
                output = torch.max(torch.min(output, self.range_up), self.range_low)
        else:
            output = input

        return output

    def parameters(self, recurse=True):
        return self.randconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.randconv.trainable_parameters()

    def whiten(self, input):
        return (input - self.data_mean) / self.data_std

    def dewhiten(self, input):
        return input * self.data_std + self.data_mean

    def randomize(self):
        self.randconv.randomize()

        if self.mixing:
            self.alpha = random.random()

    def set_test_res_weight(self, w):
        self.res_test_weight = w

class ProRandConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rand_bias=True,
                 max_L=10,
                  **kwargs):
        super(ProRandConvBlock, self).__init__()

        # super(ProRandConvBlock, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=rand_bias, **kwargs)
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, padding=1)
        self.max_L= max_L
        self.rand_bias = rand_bias
        # self.weight = deform_conv.weight
        # self.bias = deform_conv.bias

    def randomize(self):
        def gaussian_kernel(kernel_size=3):
            # 1D Gaussian
            dist = torch.distributions.uniform.Uniform(torch.tensor([1e-8]), torch.tensor([1.0]))
            sigma = dist.sample()
            x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            x = torch.exp(-x**2 / (2 * sigma**2))
            x = x / x.sum()
            # 2D Gaussian kernel
            kernel_2d = x[:, None] * x[None, :]
            return kernel_2d
        def apply_gaussian_smoothing(conv_weight, kernel_size=3):
            kernel = gaussian_kernel(kernel_size).to(conv_weight.device)
            kernel = kernel.expand(conv_weight.size(1), 1, kernel_size, kernel_size)

            smoothed_weight = F.conv2d(conv_weight, kernel, padding=kernel_size//2, groups=conv_weight.size(1))
            return smoothed_weight

        new_weight = torch.zeros_like(self.deform_conv.weight)
        with torch.no_grad():
            nn.init.kaiming_uniform_(new_weight, nonlinearity='conv2d')
            new_weight = apply_gaussian_smoothing(new_weight)
        self.weight = nn.Parameter(new_weight.detach())

        if self.rand_bias:
            # new_bias = self.bias.clone().detach()
            new_bias = torch.zeros_like(self.deform_conv.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            self.bias = nn.Parameter(new_bias)

        self.deform_conv.weight = self.weight
        self.deform_conv.bias = self.bias

        new_offset = torch.zeros_like(self.offset)
        # [batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]
        with torch.no_grad():
            dist = torch.distributions.uniform.Uniform(torch.tensor([1e-8]), torch.tensor([0.5]))
            sigma = dist.sample()
            nn.init.normal_(new_offset, mean=0.0, std=sigma.item())
        self.offset = nn.Parameter(new_offset.detach())

        self.gamma = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.5])).sample()
        self.beta = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.5])).sample()

    def forward(self, input):
        B, C, H, W = input.size()
        self.offset = torch.zeros((B, ))
        for l in range(self.max_L):
            self.randomize()
            out = self.deform_conv(input, self.offset)

        return out
