import os
import numpy as np
import torch

from torch.utils.data import Dataset
from skimage import color, io


class ImagesDateset(Dataset):
    """Custom dataset for loading and pre-processing images."""
    
    def __init__(self, img_dir, all2mem=True):
        """Initializes the dataset and loads images. 
        
        By default images are loaded to memory in
        a lazy manner i.e when one needs to get it.
        For now, all the photos must be .jpg.

        Args:
            img_dir: a directory from which images are loaded
            all2mem: if set to True, then all images
                will be read to memory at once
        """
        self.img_dir = img_dir
        self.all2mem = all2mem
        self.img_names = [file for file in os.listdir(self.img_dir)]
        
        # TODO(Przemek) In future, allow multiple image extensions.
        assert all([img.endswith('.jpg') for img in self.img_names])
        
        if self.all2mem:
            self.images = [io.imread(os.path.join(self.img_dir, img)) 
                           for img in self.img_names]
        
    
    def __len__(self):
        return len(self.img_names)
    
   
    def __getitem__(self, idx):
        """Gets an image in LAB color space.

        Returns:
            Returns a tuple (L, ab, name), where:
                L: stands for lightness - it's the net input
                ab: is chrominance - something that the net learns
                name: image filename
            Both L and ab are torch.tesnsor
        """
        
        if self.all2mem:
            image = self.images[idx]
        else:
            img_name = os.path.join(self.img_dir, self.img_names[idx])
            image = io.imread(img_name)
        
        
        assert image.shape == (224, 224, 3)
                
        img_lab = color.rgb2lab(image)
        img_lab = np.transpose(img_lab, (2, 0, 1))
        
        assert img_lab.shape == (3, 224, 224)
        
        img_lab = torch.tensor(img_lab.astype(np.float32))
        
        assert img_lab.shape == (3, 224, 224)
               
        L  = img_lab[:1,:,:]
        ab = img_lab[1:,:,:]
        
        # Normalize L and ab channels to lay in 0..1 range
        L =   L / 100.0
        ab = (ab + 128.0) / 255.0
              
        assert L.shape == (1, 224, 224)
        assert ab.shape == (2, 224, 224)
        
        return L, ab, self.img_names[idx]   