#!/usr/bin/python3

"""Handy module to manage dataset

Data is aranged as follows:

.train/
    class1/
        img1.jpg
        img2.jpg
    ..
    classn/
        imgx.jpg
        imgy.jpg
.root/         ///////////////////// make sure
    classA/
        img1.jpg
        img2.jpg
    ..
    classB/
        imgx.jpg
        imgy.jpg

Purpose of this script is to split data into train/val/test sets.
"""

import os
import shutil
from random import shuffle

def copy_imgs(imgs, src, dst):
    if not os.path.isdir(dst):
        os.makedirs(dst)

    for name in imgs:
        src_path = os.path.join(src, name)
        shutil.copy(src_path, dst)


def split_set(root, out_root, a_size, b_size=None, a_name='train', b_name='test'):
    """
    Splits files in ./root into two subsets 
        ./out_root/a_name/
        ./out_root/b_name
    """
    chosen_classes = [line.rstrip('\n') for line in open('categories100.txt', 'r')]

    print("Splitting {}/*/* into {}/{}/*/* ".format(root, out_root, a_name))

    classes_dirs = os.listdir(root)
    for c in classes_dirs:
        if c in chosen_classes:
            print("processing: {}".format(c))

            classes_paths = os.path.join(root, c)
            
            imgs = os.listdir(classes_paths)
            shuffle(imgs)

            train = imgs[:a_size]
            classes_path_out_train = os.path.join(out_root, a_name, c)
            copy_imgs(train, classes_paths, classes_path_out_train)
            
            if b_size:        
                test = imgs[a_size:a_size+b_size]
                classes_path_out_test = os.path.join(out_root, b_name, c)
                copy_imgs(test, classes_paths, classes_path_out_test)



if __name__ == "__main__":
    split_set('./data/food41-structured/train', './data/food41-new', 10, 5, 'test', 'train')
    split_set('./data/food41-structured/train', './data/food41-new', 10, None, 'val')
