# 🖌️ Automatic Image Colorization
Implementation of [_Let there be Color!_](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)
by Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa.


### First run
[Places365-Standard](http://places2.csail.mit.edu/download.html) 
dataset will be downloaded and split into _train/dev/test_ subsets.
By default only 10 arbitrary categories will be considered.

```bash
$ git clone https://github.com/kainoj/colnet.git
$ cd colnet
$ make dataset
$ make split
```

### Requirements
Python (3.6.3), pytorch (0.4.1), torchvision (0.2.1), skimage (0.14.1), numpy (1.15.2), Jupyter Notebook(4.4.0)


### Network training
Simply run:
```bash
$ python3 loader.py config/places10.yaml
```
[`places10.yaml`](./config/places10.yaml) is a sample configuration file – i.e. specifies 
total number of epoch, learning rate, output directories etc.
To see full .yaml configuration, `run python3 loader.py config/places10.yaml`


Checkpoints of models are saved on every epoch.
Training can be interrupted and resumed anytime.
Resume by executing:
```bash
$ python3 loader.py config/places10.yaml --model model.pt
```
where `model.pt` is a previously saved model checkpoint.

### Colorize!
Choose the most favourite model and hit:
```bash
$ python3 colorize.py img.jpg ./models/places.pt
```
