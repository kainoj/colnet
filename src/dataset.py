import os
import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms
from skimage import color, io
from random import randint


class HandleGrayscale(object):
    """Feeds the pipeline with 3 - channel image.
    
    All transformations below work with RGB images only.
    If a 1-channel grayscale image is given, it's converted to
    equivalent 3-channel RGB image.
    """
    def __call__(self, image):
        if len(image.shape) < 3:
            image = color.gray2rgb(image)
        return image


class RandomCrop(object):
    """Randomly crops an image to size x size."""
    
    def __init__(self, size=224):
        self.size = size
        
    def __call__(self, image):

        h, w, _ = image.shape
        assert min(h, w) >= self.size

        off_h = randint(0, h - self.size)
        off_w = randint(0, w - self.size)

        cropped = image[off_h:off_h+self.size, off_w:off_w+self.size]

        assert cropped.shape == (self.size, self.size, 3)
        return cropped

    
class Rgb2LabNorm(object):
    """Converts an RGB image to normalized image in LAB color sapce.
    
    Both (L, ab) channels are in [0, 1].
    """
    def __call__(self, image):
        assert image.shape == (224, 224, 3)
        img_lab = color.rgb2lab(image)
        img_lab[:,:,:1] = img_lab[:,:,:1] / 100.0
        img_lab[:,:,1:] = (img_lab[:,:,1:] + 128.0) / 256.0
        return img_lab
    
    
class ToTensor(object):
    """Converts an image to torch.Tensor.
    
    Note:
        Images are Height x Width x Channels format. 
        One needs to convert them to CxHxW.
    """
    def __call__(self, image):
        
        assert image.shape == (224, 224, 3)
        
        transposed =  np.transpose(image, (2, 0, 1)).astype(np.float32)
        image_tensor = torch.from_numpy(transposed)

        assert image_tensor.shape == (3, 224, 224)
        return image_tensor

    
class SplitLab(object):
    """Splits tensor LAB image to L and ab channels."""
    def __call__(self, image):
        assert image.shape == (3, 224, 224)
        L  = image[:1,:,:]
        ab = image[1:,:,:]
        return (L, ab)
    
    
    
class ImagesDateset(torchvision.datasets.ImageFolder):
    """Custom dataset for loading and pre-processing images."""

    def __init__(self, root, testing=False):
        """Initializes the dataset and loads images. 

        If testing is set, then image name is returned instead of label.

        Imges should be organized as:
        
            .root/
                class1/
                    img1.jpg
                    img2.jpg
                ..
                classn/
                    imgx.jpg
                    imgy.jpg

        Args:
            root: a directory from which images are loaded
            testing: if set to True, an image name will 
                be returned insead of label index
        """
        super().__init__(root=root, loader=io.imread)
        
        self.testing = testing

        self.composed = torchvision.transforms.Compose(
            [HandleGrayscale(), RandomCrop(224), Rgb2LabNorm(), 
             ToTensor(), SplitLab()]
        )
        
            
        
    def __getitem__(self, idx):
        """Gets an image in LAB color space.

        Returns:
            Returns a tuple (L, ab, label, name), where:
                L: stands for lightness - it's the net input
                ab: is chrominance - something that the net learns
                label: image label. If in testing mode, this is an image name.
            Both L and ab are torch.tesnsor
        """
        image, label =  super().__getitem__(idx)
        
        L, ab = self.composed(image)

        if self.testing:
            label = self.get_name(idx)

        return L, ab, label

    
    def get_name(self, idx):
        path = os.path.normpath(self.imgs[idx][0])
        name = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path))
        return label + "-" + name