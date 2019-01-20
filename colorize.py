import os
import argparse

import torch
import torchvision.transforms

import numpy as np

from skimage import io, color
from src import colnet
from src import dataset
from src import utils


def colorize(img_path, model):
    
    # Load model
    checkpoint = torch.load(model, map_location=torch.device("cpu"))
    classes = checkpoint['classes']
    net_divisor = checkpoint['net_divisor']
    num_classes = len(classes)

    net = colnet.ColNet(num_classes=num_classes, net_divisor=net_divisor)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # Image transforms
    composed_transforms = torchvision.transforms.Compose(
            [dataset.HandleGrayscale(), 
             dataset.RandomCrop(224),
             dataset.Rgb2LabNorm(), 
             dataset.ToTensor(), 
             dataset.SplitLab()]
        )
    
    # Load and process image
    img = io.imread(img_path)
    img_name = os.path.basename(img_path)

    L, ab = composed_transforms(img)
    L_tensor = torch.from_numpy(np.expand_dims(L, axis=0))
    
    # Colorize
    softmax = torch.nn.Softmax(dim=1)
    net.eval()
    with torch.no_grad():
        ab_out, predicted = net(L_tensor)
        img_colorized = utils.net_out2rgb(L, ab_out[0])

        colorized_img_name = "colorized-" + img_name
        io.imsave(colorized_img_name, img_colorized)

        print("\nSaved image to: {}\n".format(colorized_img_name))


        sm = softmax(predicted)
        probs = sm[0].numpy()

        probs_and_classes = sorted(zip(probs, classes), key=lambda x: x[0], reverse=True)


        print("Predicted labels: \n")
        for p, c in probs_and_classes[:10]:
            print("{:>7.2f}% \t{}".format(p*100.0, c))

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to colorize a photo")
    parser.add_argument('image', help="Path to the image. RGB one will be converted to grayscale")
    parser.add_argument('model', help="Path a *.pt model")
    args = parser.parse_args()
    
    print("[Warrning] Only 224x224 images are supported. Otherwise an image will be randomly cropped")
    
    
    colorize(args.image, args.model)
                 





    
    
    
    
    
    