import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from skimage import io

from .colnet import ColNet
from .dataset import ImagesDateset
from .utils import net_out2rgb


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
                 model_checkpoint=None,
                 models_dir='./model/',
                 img_out_dir='./out',
                 num_workers=4):
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
            models_dir: directory to which models are saved. DEFAULT: ./model
            img_out_dir: a directory where colorized
                images are saved. DEFAULT: ./out
        """
        self.img_dir_train = img_dir_train
        self.img_dir_val = img_dir_val
        self.img_dir_test = img_dir_test
        self.net_size = net_size
        
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
              os.makedirs(self.models_dir)

        self.img_out_dir = img_out_dir
        if not os.path.exists(self.img_out_dir):
              os.makedirs(self.img_out_dir)


        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")
        
        self.net = ColNet(net_size=net_size)
        self.net.to(self.device)
        print("Using {}\n".format(self.device))
        
        self.start_epoch = start_epoch
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.loss_history = { "train": [], "val":[] }
        
        self.mse = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        
        
        if model_checkpoint:
            self.load_checkpoint(model_checkpoint)


        self.trainset = ImagesDateset(self.img_dir_train)
        self.trainloader = DataLoader(self.trainset, batch_size=self.BATCH_SIZE, 
                                      shuffle=True, num_workers=num_workers)

        self.testset = ImagesDateset(self.img_dir_test, testing=True)
        self.testloader = DataLoader(self.testset, batch_size=self.BATCH_SIZE,
                                     shuffle=False, num_workers=num_workers)

        self.devset = ImagesDateset(self.img_dir_val)
        self.devloader = DataLoader(self.devset, batch_size=self.BATCH_SIZE,
                                    shuffle=False, num_workers=num_workers)


        self.current_model_name = model_checkpoint  


    def train(self, epoch):
        """One epoch network training"""

        epoch_loss = 0.0
        
        # Turn train mode on
        self.net.train() 

        for batch_idx, train_data in enumerate(self.trainloader):

            L, ab, _ = train_data
            L, ab = L.to(self.device), ab.to(self.device)

            self.optimizer.zero_grad()
            ab_out = self.net(L)
            
            
            assert ab.shape == ab_out.shape
            
            loss = self.mse(ab, ab_out)
            loss.backward()
            self.optimizer.step()
        
            batch_loss = loss.item()
            
            print('[Epoch {:>2} / {} | Batch: {:>2} / {}] loss: {:>10.3f}'
                .format(epoch+1, self.EPOCHS, batch_idx + 1, len(self.trainloader), batch_loss))
            epoch_loss += batch_loss
            
        # Epoch loss = mean loss over all batches
        # length of trainloader indicates number of batches
        epoch_loss /= len(self.trainloader)
        self.loss_history['train'].append(epoch_loss)

        print("Epoch loss: {:.5f}".format(epoch_loss))

    def validate(self, epoch):
        """One epoch validation on a dev set"""

        print("\nValidating...")
        dev_loss = 0.0

        # Turn eval mode on
        self.net.eval()
        with torch.no_grad():
            
            for batch_idx, dev_data in enumerate(self.devloader):

                L_dev, ab_dev, _ = dev_data
                L_dev, ab_dev = L_dev.to(self.device), ab_dev.to(self.device)

                ab_dev_output = self.net(L_dev)

                assert ab_dev.shape == ab_dev_output.shape
                
                dev_batch_loss = self.mse(ab_dev_output, ab_dev)
                dev_loss += dev_batch_loss

                print("[Validation] [Batch {:>2} / {}] dev loss: {:>10.3f}"
                    .format(batch_idx+1, len(self.devloader), dev_batch_loss))
                
                
                
        print("Dev loss {:.5f}".format(dev_loss.item()/len(self.devloader)))
        
        dev_loss /= len(self.devloader)
        self.loss_history['val'].append(dev_loss)


    def test(self, model_dir=None):
        """Tests network on a test set.
        
        Saves all pics to a predefined directory (self.img_out_dir)
        """

        if model_dir is None:
            model_dir = self.current_model_name

        print("Make sure you're using up to date model!!!")    
        print("Colorizing {} using {}\n".format(self.img_dir_test, model_dir))


        self.load_checkpoint(model_dir)
        self.net.to(self.device)

        # Switch to evaluation mode
        self.net.eval()

        with torch.no_grad():
            for batch_no, data in enumerate(self.testloader):
                
                print("Processing batch {} / {}"
                      .format(batch_no + 1, len(self.testloader)))
                
                L, _, name = data
                L = L.to(self.device)
                ab_outputs = self.net(L)
                
                L = L.to(torch.device("cpu"))
                ab_outputs = ab_outputs.to(torch.device("cpu"))
                
                for i in range(L.shape[0]):
                    img = net_out2rgb(L[i], ab_outputs[i])
                    io.imsave(os.path.join(self.img_out_dir, name[i]), img)
                
        print("Saved all photos to " + self.img_out_dir)


    def save_checkpoint(self, epoch):
        """Saves a checkpoint of the model to a file."""
        path = self.models_dir
        fname = "colnet{}-{}.pt".format(time.strftime("%y%m%d-%H-%M-%S"), epoch)
        full_path = os.path.join(path, fname)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.loss_history,
            'net_size': self.net_size    
        }, full_path)        

        self.current_model_name = full_path
        print('\nsaved model to {}\n'.format(full_path))


    def load_checkpoint(self, model_checkpoint):
        """Load a checkpoint from a given path.
        
        Args:
            model_checkpoint: path to the checkpoint.
        """
        print("Resuming training of: " + model_checkpoint)
        checkpoint = torch.load(model_checkpoint, map_location=torch.device("cpu"))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['losses']
        self.start_epoch = checkpoint['epoch'] + 1 
        self.net_size = checkpoint['net_size'] 
        self.current_model_name = model_checkpoint


    def run(self):
        """Runs both training and validating."""
        for epoch in range(self.start_epoch, self.EPOCHS):
            print("{2}\nEpoch {0} / {1}\n{2}"
                  .format(epoch + 1, self.EPOCHS, '-'*47))
            self.train(epoch)
            self.validate(epoch)
            self.save_checkpoint(epoch)
        print('\nFinished Training.\n')


    def info(self):
        print("{0} Training environment info {0}\n".format("-"*13))

        print("Training starts from epoch: {}".format(self.start_epoch))
        print("Total number of epochs:     {}".format(self.EPOCHS))
        print("ColNet parameters are devided by: {}".format(self.net_size))
        print("Batch size:  {}".format(self.BATCH_SIZE))
        print("Used devide: {}".format(self.device))
        print()

        if self.current_model_name:
            print("Current model name:      " + self.current_model_name)

        print("Training data directory: " + self.img_dir_train)
        print("Validate data directory: " + self.img_dir_val)
        print("Testing data directory:  " + self.img_dir_test)
        print("Models are saved to:     " + self.models_dir)
        print("Colorized images are saved to: " + self.img_out_dir)
        print("-" * 53 + "\n")



if __name__ == "__main__":
    print("Hello, have a great day!")