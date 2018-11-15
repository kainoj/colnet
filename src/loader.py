"""This module is responsible for reading configuration from YAML file.


Structure of YAML file:
    TODO(Przemek) model: (optional) snapshot of a net state. 
        If given, training will go on based on rest of parameters.
        If not given, a fresh instance of a net will be created.
    epochs: total number of epoches for model yo run.
    batch_size: batch size for train, test and dev sets.
    net_size: (optional) divisor of net optput sizes. DEFAULT: 1.
    learning_rate: (optional) learning rate of a net. DEFAULT: 0.0001.

    img_dir_train: name of directory containing images for TRAINING.
    img_dir_val: name of directory containing images for VALIDATING.
    img_dir_test: name of directory containing images for TESTING.
"""

import yaml
from trainer import Training
from colnet import ColNet


def load_config(config_file):
    """Loads config from YAML file

    Args:
        config_file: path to config file
    Returns:
        Instance of ColNet
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

        # TODO(Przemek): implement loading a model

        train = Training(batch_size=y['batch_size'],
                         epochs=y['epochs'],
                         img_dir_train=y['img_dir_train'],
                         img_dir_val=y['img_dir_val'],
                         img_dir_test=y['img_dir_val'],
                         net_size=net_size,
                         learning_rate=learning_rate)

        return train



if __name__ == "__main__":
    
    t = load_config('../config/test0.yaml')
    t.run()
    