import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def Conv2d(in_ch, out_ch, stride, kernel_size=3, padding=1):
        """Returns an instance of nn.Conv2d"""
        return nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                         stride=stride, kernel_size=kernel_size, padding=padding)


class LowLevelFeatures(nn.Module):
    """Low-Level Features Network"""

    def __init__(self, net_divisor=1):
        super(LowLevelFeatures, self).__init__()

        ksize = np.array([1, 64, 128, 128, 256, 256, 512]) // net_divisor
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

    def __init__(self, net_divisor=1):
        super(MidLevelFeatures, self).__init__()

        ksize = np.array([512, 512, 256]) // net_divisor

        self.conv7 = Conv2d(ksize[0], ksize[1], 1)
        self.conv8 = Conv2d(ksize[1], ksize[2], 1)

    def forward(self, x):
        out = F.relu(self.conv7(x))
        out = F.relu(self.conv8(out))
        return out


class GlobalFeatures(nn.Module):
    """Global Features Network"""

    def __init__(self, net_divisor=1):
        super(GlobalFeatures, self).__init__()

        ksize = np.array([512, 1024, 512, 256]) // net_divisor
        self.ksize0 = ksize[0]

        self.conv1 = Conv2d(ksize[0], ksize[0], 2)
        self.conv2 = Conv2d(ksize[0], ksize[0], 1)
        self.conv3 = Conv2d(ksize[0], ksize[0], 2)
        self.conv4 = Conv2d(ksize[0], ksize[0], 1)
        self.fc1 = nn.Linear(7*7*ksize[0], ksize[1])
        self.fc2 = nn.Linear(ksize[1], ksize[2])
        self.fc3 = nn.Linear(ksize[2], ksize[3])

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = y.view(-1, 7*7*self.ksize0)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        
        # Branching
        out = y
        classification_in = y

        out = F.relu(self.fc3(out))

        return out, classification_in

class ColorizationNetwork(nn.Module):
    """Colorizaion Network"""

    def __init__(self, net_divisor=1):
        super(ColorizationNetwork, self).__init__()

        ksize = np.array([256, 128, 64, 64, 32]) // net_divisor

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


class ClassNet(nn.Module):
    """Classification Network Class"""

    def __init__(self, num_classes, net_divisor=1):
        super(ClassNet, self).__init__()
        
        self.num_classes = num_classes
        ksize = np.array([512, 256]) // net_divisor

        self.fc1 = nn.Linear(ksize[0], ksize[1])
        self.fc2 = nn.Linear(ksize[1], num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out





class ColNet(nn.Module):
    """Colorization network class"""

    def __init__(self, num_classes, net_divisor=1):
        """Initializes the network.

        Args:
            net_divisor - divisor of net output sizes. Useful for debugging.
        """
        super(ColNet, self).__init__()

        self.net_divisor = net_divisor

        self.conv_fuse = Conv2d(512 // net_divisor, 256 // net_divisor, 1, kernel_size=1, padding=0)

        self.low = LowLevelFeatures(net_divisor)
        self.mid = MidLevelFeatures(net_divisor)
        self.classifier = ClassNet(num_classes, net_divisor)
        self.glob = GlobalFeatures(net_divisor)
        self.col = ColorizationNetwork(net_divisor)



    def fusion_layer(self, mid_out, glob_out):
        h = mid_out.shape[2]  # Height of a picture  
        w = mid_out.shape[3]  # Width of a picture
        
        glob_stack2d = torch.stack(tuple(glob_out for _ in range(w)), 1)
        glob_stack3d = torch.stack(tuple(glob_stack2d for _ in range(h)), 1)
        glob_stack3d = glob_stack3d.permute(0, 3, 1, 2)

        # 'Merge' two volumes
        stack_volume = torch.cat((mid_out, glob_stack3d), 1)

        out = F.relu(self.conv_fuse(stack_volume))
        return out


    def forward(self, x):
        # Low level
        low_out = self.low(x)
        
        # Net branch         
        mid_out = low_out
        glob_out = low_out

        # Mid level
        mid_out = self.mid(mid_out)

        # Global
        glob_out, classification_in = self.glob(glob_out)

        # Classification
        classification_out = self.classifier(classification_in)

        # Fusion layer
        out = self.fusion_layer(mid_out, glob_out)

        # Colorization Net
        out = self.col(out)
        
        return out, classification_out
        