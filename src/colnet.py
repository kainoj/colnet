import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ColNet(nn.Module):
    """Colorization network class"""

    def Conv2d(self, in_ch, out_ch, stride):
        """Returns an instance of nn.Conv2d"""
        return nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                         stride=stride, kernel_size=3, padding=1)


    def __init__(self, net_size=1):
        """Initializes the network.

        Args:
            net_size - divisor of net output sizes. Useful for debugging.
        """
        super(ColNet, self).__init__()

        self.net_size = net_size


        ksize = np.array([1, 64, 128, 128, 256, 256, 512,
                          512, 256, 128, 64, 64, 32]) // self.net_size
        ksize[0] = 1
        
        # 'Low-level features'
        self.conv1 = self.Conv2d(1, ksize[1], 2)
        self.conv2 = self.Conv2d(ksize[1], ksize[2], 1)
        self.conv3 = self.Conv2d(ksize[2], ksize[3], 2)
        self.conv4 = self.Conv2d(ksize[3], ksize[4], 1)
        self.conv5 = self.Conv2d(ksize[4], ksize[5], 2)
        self.conv6 = self.Conv2d(ksize[5], ksize[6], 1)
        
        # 'Mid-level fetures'
        self.conv7 = self.Conv2d(ksize[6], ksize[7], 1)
        self.conv8 = self.Conv2d(ksize[7], ksize[8], 1)
        
        # 'Colorization network'
        self.conv9 = self.Conv2d(ksize[8], ksize[9], 1)
        
        # Here comes upsample #1
        
        self.conv10 = self.Conv2d(ksize[9], ksize[10], 1)
        self.conv11 = self.Conv2d(ksize[10], ksize[11], 1)
        
        # Here comes upsample #2        
        
        self.conv12 = self.Conv2d(ksize[11], ksize[12], 1)
        self.conv13 = self.Conv2d(ksize[12], 2, 1)
        
        
    def forward(self, x):
        
        # Low level
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        
        # y = out
        # z = out
        # y → mid level
        # z → global features 
        
        # Mid level
    
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        
        # TODO(Przemek) generalize asserts so they work with every net shapes
        # assert out.shape[1:] == (256, 28, 28), "おわり： mid level"
        

        # Colorization Net
        out = F.relu(self.conv9(out))
        
        # assert out.shape[1:] == (128, 28, 28), "おわり： conv9"
        
        out = nn.functional.interpolate(input=out, scale_factor=2)

        # assert out.shape[1:] == (128, 56, 56), "おわり： upsample1"
    
        out = F.relu(self.conv10(out))
        out = F.relu(self.conv11(out))
        
        out = nn.functional.interpolate(input=out, scale_factor=2)


        out = F.relu(self.conv12(out))
        out = torch.sigmoid(self.conv13(out))
        
        out = nn.functional.interpolate(input=out, scale_factor=2)
        
        # assert out.shape[1:] == (2, 224, 224)
        
        return out
        