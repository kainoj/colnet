import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from skimage import io

from colnet import ColNet
from dataset import ImagesDateset
from utils import net_out2rgb


class Training:
    """Trains model based on given hyperparameterms"""

    def __init__(self,
                 batch_size,
                 epochs,
                 img_dir_train, 
                 img_dir_val,
                 img_dir_test,
                 start_epoch=0,
                 net_size=1,
                 learning_rate=0.0001,
                 model_checkpoint=None):
        """Initializes training environment

        Args:
            batch_size: size of a batch
            epoches: number of epoches to run
            img_dir_train: name of directory containing images for TRAINING
            img_dir_val: name of directory containing images for VALIDATING
            img_dir_test: name of directory containing images for TESTING
            start_epoch: epoch to start training with. Default: 0
            net_size: divisor og the net output sizes. Default: 1
            learning_rate: alpha parameter of GD/ADAM. Default: 0.0001
            model_checkpoint: a path to a previously saved model. 
                Training will resume. Defaut: None
        """
        self.img_dir_train = img_dir_train
        self.img_dir_val = img_dir_val
        self.img_dir_test = img_dir_test
        self.net_size = net_size

        self.net = ColNet(net_size=net_size)
        self.mse = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.start_epoch = start_epoch
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

        if model_checkpoint:
            self.load_checkpoint(model_checkpoint)


        self.trainset = ImagesDateset(self.img_dir_train, all2mem=True)
        self.trainloader = DataLoader(self.trainset, batch_size=self.BATCH_SIZE, 
                                      shuffle=True, num_workers=4)

        self.testset = ImagesDateset(self.img_dir_test, all2mem=True)
        self.testloader = DataLoader(self.testset, batch_size=self.BATCH_SIZE,
                                     shuffle=False, num_workers=4)

        self.devset = ImagesDateset(self.img_dir_val, all2mem=True)
        self.devloader = DataLoader(self.devset, batch_size=self.BATCH_SIZE,
                                    shuffle=False, num_workers=4)


        # We'll keep track on all models 
        # names that were saved on traing
        self.model_names_history = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")
        print("Using {}\n".format(self.device))
        self.net.to(self.device)


    def train(self, epoch):
        """One epoch network training"""

        epoch_loss = 0.0
        
        print("Epoch {} / {}".format(epoch + 1, self.EPOCHS))
        
        # Turn train mode on
        self.net.train() 

        for batch_idx, train_data in enumerate(self.trainloader):

            L, ab, _ = train_data
            
            self.optimizer.zero_grad()
            ab_out = self.net(L)
            
            
            assert ab.shape == ab_out.shape
            
            loss = self.mse(ab, ab_out)
            loss.backward()
            self.optimizer.step()
        
            batch_loss = loss.item()
            
            print('[{} / {}] batch loss: {:.3f}'
                .format(batch_idx + 1, len(self.trainloader), batch_loss))
            epoch_loss += batch_loss
            

    def validate(self, epoch):
        """One epoch validation on a dev set"""

        print("\nValidating...")
    
        # Turn eval mode on
        self.net.eval()
        with torch.no_grad():
            
            dev_loss = 0.0
            
            for batch_idx, dev_data in enumerate(self.devloader):

                L_dev, ab_dev, _ = dev_data
                L_dev, ab_dev = L_dev.to(self.device), ab_dev.to(self.device)

                ab_dev_output = self.net(L_dev)

                assert ab_dev.shape == ab_dev_output.shape
                
                dev_batch_loss = self.mse(ab_dev_output, ab_dev)
                dev_loss += dev_batch_loss

                print("[{} / {}] dev batch loss: {}"
                    .format(batch_idx+1, len(self.devloader), dev_batch_loss))
                
                
                
            print("sum of dev losses {}".format(dev_loss.item()))
            print("mean of dev losses {}"
                  .format(dev_loss.item()/len(self.devloader)))


    def test(self, model_dir=None):
        """Tests network on a test set.
        
        Saves all pics to ../out/
        """

        if model_dir is None:
            model_dir = self.model_names_history[-1]

        print("Make sure you're using up to date model!!!")    
        print("Colorizing {} using {}\n".format(self.img_dir_test, model_dir))


        self.load_checkpoint(model_dir)
        
        # Switch to evaluation mode
        self.net.eval()

        with torch.no_grad():
            for batch_no, data in enumerate(self.testloader):
                
                print("Processing batch {} / {}"
                      .format(batch_no + 1, len(self.testloader)))
                
                L, _, name = data
                ab_outputs = self.net(L)
                
                for i in range(L.shape[0]):
                    print("processing: {}".format(name[i]))
                    img = net_out2rgb(L[i], ab_outputs[i])
                    io.imsave(os.path.join("../out/", name[i]), img)
                
        print("Saved all photos to ../out/")


    def save_checkpoint(self, epoch):
        """Saves a checkpoint of the model to a file."""
        path = "../model/"
        fname = "colnet{}-{}.pt".format(time.strftime("%y%m%d-%H-%M-%S"), epoch)
        full_path = os.path.join(path, fname)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()         
        }, full_path)        

        self.model_names_history.append(full_path)
        print('\nsaved model to {}'.format(full_path))


    def load_checkpoint(self, model_checkpoint):
        """Load a checkpoint from a given path.
        
        Args:
            model_checkpoint: path to the checkpoint.
        """
        print("Resuming training of: " + model_checkpoint)
        checkpoint = torch.load(model_checkpoint)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1 


    def run(self):
        """Runs both training and validating."""
        for epoch in range(self.start_epoch, self.EPOCHS):
            self.train(epoch)
            self.validate(epoch)
            self.save_checkpoint(epoch)
        print('Finished Training.')


if __name__ == "__main__":
    img_dir_train = '../data/food41-120-train/'
    img_dir_test  = '../data/food41-120-train/'
    img_dir_dev   = '../data/food41-120-test/'   

    train = Training(8, 120, img_dir_train=img_dir_train,
                     img_dir_test=img_dir_test, 
                     img_dir_val=img_dir_dev,
                     net_size=4)

    train.run()
    train.test()