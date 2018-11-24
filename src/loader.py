"""This module is responsible for reading configuration from YAML file.


Structure of YAML file:
    model_checkpoint: (optional) path to a checkpoint of a net state. 
        If given, training resume on based on rest of parameters.
    epochs: total number of epoches for model yo run.
    batch_size: batch size for train, test and dev sets.
    net_size: (optional) divisor of net optput sizes. DEFAULT: 1.
    learning_rate: (optional) learning rate of a net. DEFAULT: 0.0001.

    img_dir_train: name of directory containing images for TRAINING.
    img_dir_val: name of directory containing images for VALIDATING.
    img_dir_test: name of directory containing images for TESTING.
"""

import yaml
import argparse
from trainer import Training
from colnet import ColNet


def load_config(config_file, model_checkpoint=None):
    """Loads config from YAML file

    Args:
        config_file: path to config file
    Returns:
        Instance of Training environment
    """

    # Default parameters
    net_size = 1
    learning_rate = 0.0001

    with open(config_file, 'r') as conf:
        y = yaml.load(conf)

        if 'net_size' in y:
            net_size = y['net_size']
        
        if 'learning_rate' in y:
            learning_rate = y['learning_rate']

        if 'model_checkpoint' in y:
            model_checkpoint = y['model_checkpoint']


        train = Training(batch_size=y['batch_size'],
                         epochs=y['epochs'],
                         img_dir_train=y['img_dir_train'],
                         img_dir_val=y['img_dir_val'],
                         img_dir_test=y['img_dir_val'],
                         net_size=net_size,
                         learning_rate=learning_rate,
                         model_checkpoint=model_checkpoint)

        return train



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Loads network configuration.')
    parser.add_argument('config', metavar='config', help='path to .yaml config file')
    args = parser.parse_args()

    t = load_config(args.config)
    t.run()
    t.test()
    