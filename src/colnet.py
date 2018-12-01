import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Conv2d(in_ch, out_ch, stride):
        """Returns an instance of nn.Conv2d"""
        return nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                         stride=stride, kernel_size=3, padding=1)


class LowLevelFeatures(nn.Module):
    """Low-Level Features Network"""

    def __init__(self, net_size=1):
        super(LowLevelFeatures, self).__init__()

        ksize = np.array([1, 64, 128, 128, 256, 256, 512]) // net_size
        ksize[0] = 1

        self.conv1 = Conv2d(1, ksize[1], 2)
        self.conv2 = Conv2d(ksize[1], ksize[2], 1)
        self.conv3 = Conv2d(ksize[2], ksize[3], 2)
        self.conv4 = Conv2d(ksize[3], ksize[4], 1)
        self.conv5 = Conv2d(ksize[4], ksize[5], 2)
        self.conv6 = Conv2d(ksize[5], ksize[6], 1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        return out


class MidLevelFeatures(nn.Module):
    """Mid-Level Features Network"""

    def __init__(self, net_size=1):
        super(MidLevelFeatures, self).__init__()

        ksize = np.array([512, 512, 256]) // net_size

        self.conv7 = Conv2d(ksize[0], ksize[1], 1)
        self.conv8 = Conv2d(ksize[1], ksize[2], 1)

    def forward(self, x):
        out = F.relu(self.conv7(x))
        out = F.relu(self.conv8(out))
        return out


class ColorizationNetwork(nn.Module):
    """Colorizaion Network"""

    def __init__(self, net_size=1):
        super(ColorizationNetwork, self).__init__()

        ksize = np.array([256, 128, 64, 64, 32]) // net_size

        self.conv9 = Conv2d(ksize[0], ksize[1], 1)
        
        # Here comes upsample #1
        
        self.conv10 = Conv2d(ksize[1], ksize[2], 1)
        self.conv11 = Conv2d(ksize[2], ksize[3], 1)
        
        # Here comes upsample #2        
        
        self.conv12 = Conv2d(ksize[3], ksize[4], 1)
        self.conv13 = Conv2d(ksize[4], 2, 1)
    
    def forward(self, x):
        out = F.relu(self.conv9(x))

        # Upsample #1        
        out = nn.functional.interpolate(input=out, scale_factor=2)

        out = F.relu(self.conv10(out))
        out = F.relu(self.conv11(out))
        
        # Upsample #2
        out = nn.functional.interpolate(input=out, scale_factor=2)

        out = F.relu(self.conv12(out))
        out = torch.sigmoid(self.conv13(out))
        
        # Upsample #3
        out = nn.functional.interpolate(input=out, scale_factor=2)
        
        return out



class ColNet(nn.Module):
    """Colorization network class"""

    def __init__(self, net_size=1):
        """Initializes the network.

        Args:
            net_size - divisor of net output sizes. Useful for debugging.
        """
        super(ColNet, self).__init__()

        self.net_size = net_size

        self.low = LowLevelFeatures(net_size)
        self.mid = MidLevelFeatures(net_size)
        self.col = ColorizationNetwork(net_size)


    def forward(self, x):
        # Low level
        out = self.low(x)
        
        # y = out
        # z = out
        # y → mid level
        # z → global features
         
        # Mid level
        out = self.mid(out)

        # Colorization Net
        out = self.col(out)
        
        return out
        