import numpy as np
from skimage import color


def net_out2rgb(L, ab_out):
    """Translates the net output back to an image.

    More specifically: unnormalizes both L and ab_out channels, stacks them
    into an image in LAB color space and converts back to RGB.
  
    Args:
        L: original L channel of an image
        ab_out: ab channel which was learnt by the network
    
    Retruns: 
        3 channel RGB image
    """
    # Convert to numpy and unnnormalize
    L = (L.numpy() * 100.0).astype(np.int8)
    ab_out = (np.floor(ab_out.numpy() * 255.0) - 128.0).astype(np.int8)
    
    
    # L and ab_out are tenosr i.e. are of shape of
    # Height x Width x Channels
    # We need to transpose axis back to HxWxC
    L = L.transpose((1, 2, 0))
    ab_out = ab_out.transpose((1, 2, 0))

    # Stack layers  
    img_stack = np.dstack((L, ab_out))
    
    return color.lab2rgb(img_stack)