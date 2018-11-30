#!/usr/bin/python3

"""Handy module to manage dataset."""

example = """
Short example. Think of 'a_name' as a 'train' and 'b_name' as a 'test'.

→ categories.txt: (newline separated).
    cat1
    cat3 

→ Dataset:
    root/a_name/
        cat1/
        cat2/
        cat3/
    root/b_name/
        cat1/
        cat2/
        cat3/

→ This script will split root/ into:
    out/a_name/
        cat1/
        cat3/
    out/b_name/
        cat1/
        cat3/

Files in out/a_name and out/b_name are unique.
Each direcotry (class) in out/a_name has a_size files.
Each direcotry (class) in out/b_name has b_size files.
"""

import os
import sys
import shutil
import argparse
from random import shuffle

def copy_imgs(imgs, src, dst):
    if not os.path.isdir(dst):
        os.makedirs(dst)

    for name in imgs:
        src_path = os.path.join(src, name)
        shutil.copy(src_path, dst)


def split_set(root, out_root,
              a_size, b_size=None, 
              a_name='train', b_name='test', 
              chosen_categories_src=None):
    """
    Splits files in ./root into two subsets 
        ./out_root/a_name/
        ./out_root/b_name
    """
    chosen_classes = [line.rstrip('\n') for line in open(chosen_categories_src, 'r')]

    print("Splitting into {}{}".format(out_root, a_name))
    if b_size:
        print("Splitting into {}{}".format(out_root, b_name))

    a_size = int(a_size)

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
                b_size = int(b_size)
                test = imgs[a_size:a_size+b_size]
                classes_path_out_test = os.path.join(out_root, b_name, c)
                copy_imgs(test, classes_paths, classes_path_out_test)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Splits dataset.\n' + example,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='Directory of dataset root')
    parser.add_argument('out', help='Output directorys')
    parser.add_argument('classes', help='File; newline separated list of ' + \
                                      'classes that are used on split')
    parser.add_argument('aname', help='Name of set a')
    parser.add_argument('asize', help='Number of files in each category in set a')


    parser.add_argument('--bname', help='Name of set b', default=None)
    parser.add_argument('--bsize', help='Number of files in each category in set b', default=None)

    args = parser.parse_args()

    split_set(args.root, args.out, args.asize, args.bsize, args.aname, args.bname, args.classes)