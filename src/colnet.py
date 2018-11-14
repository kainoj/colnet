import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ColNet(nn.Module):
    """Colorization network class"""

    
    def __init__(self, net_size=1):
        """Initializes the network.

        Args:
            net_size - divisor of net output sizes. Useful for debugging.
        """
        super(ColNet, self).__init__()

        self.net_size = net_size


        ksize = np.array( [1, 64, 128, 128, 256, 256, 512, 512, 256, 128, 64, 64, 32] ) // self.net_size
        ksize[0] = 1
        
        # 'Low-level features'
        self.conv1 = nn.Conv2d(in_channels=1,        out_channels=ksize[1], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=ksize[1], out_channels=ksize[2], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=ksize[2], out_channels=ksize[3], kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=ksize[3], out_channels=ksize[4], kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=ksize[4], out_channels=ksize[5], kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=ksize[5], out_channels=ksize[6], kernel_size=3, stride=1, padding=1)
        
        # 'Mid-level fetures'
        self.conv7 = nn.Conv2d(in_channels=ksize[6], out_channels=ksize[7], kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=ksize[7], out_channels=ksize[8], kernel_size=3, stride=1, padding=1)
        
        # 'Colorization network'
        self.conv9 = nn.Conv2d(in_channels=ksize[8], out_channels=ksize[9], kernel_size=3, stride=1, padding=1)
        
        # Here comes upsample #1
        
        self.conv10 = nn.Conv2d(in_channels=ksize[9], out_channels=ksize[10], kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=ksize[10],out_channels=ksize[11], kernel_size=3, stride=1, padding=1)
        
        # Here comes upsample #2        
        
        self.conv12 = nn.Conv2d(in_channels=ksize[11], out_channels=ksize[12], kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=ksize[12], out_channels=2, kernel_size=3, stride=1, padding=1)
        
        
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
        
        out = nn.functional.interpolate(input=out, scale_factor=2, mode='nearest')

        # assert out.shape[1:] == (128, 56, 56), "おわり： upsample1"
    
        out = F.relu(self.conv10(out))
        out = F.relu(self.conv11(out))
        
        out = nn.functional.interpolate(input=out, scale_factor=2, mode='nearest')


        out = F.relu(self.conv12(out))
        out = torch.sigmoid(self.conv13(out))
        
        out = nn.functional.interpolate(input=out, scale_factor=2, mode='nearest')
        
        # assert out.shape[1:] == (2, 224, 224)
        
        return out
        