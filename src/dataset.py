import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms
from skimage import color, io


class RandomCrop(object):
    """Randomly crops an image to size x size."""
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        cropped = image  # TODO(Przemek): implement random cropping
        assert cropped.shape == (224, 224, 3)
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

    def __init__(self, root, all2mem=False):
        """Initializes the dataset and loads images. 
        
        By default images are loaded to memory in
        a lazy manner i.e when one needs to get it.

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
            all2mem: if set to True, then all images
                will be read to memory at once
        """
        super().__init__(root=root, loader=io.imread)
        
        if all2mem:
            print("[WARNING] all2mem temporarily disabled")
            
        self.composed = torchvision.transforms.Compose(
            [RandomCrop(224), Rgb2LabNorm(), ToTensor(), SplitLab()]
        )
        
            
        
    def __getitem__(self, idx):
        """Gets an image in LAB color space.

        Returns:
            Returns a tuple (L, ab, label), where:
                L: stands for lightness - it's the net input
                ab: is chrominance - something that the net learns
                label: image label
            Both L and ab are torch.tesnsor
        """
        image, label =  super().__getitem__(idx)
        
        L, ab = self.composed(image)

        # TODO(Przemek) 
        label = str(idx) + "-" + str(label) + "-todo.jpg"
        return L, ab, label