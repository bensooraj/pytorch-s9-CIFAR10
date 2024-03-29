# PyTorch CIFAR10 Dataset Deep Learning Experiment

## 1. Introduction
In continuation to [bensooraj/pytorch-s9-CIFAR10](https://github.com/bensooraj/pytorch-s9-CIFAR10/tree/main), this project demonstrates the use of,
1. Depthwise-separable convolution to reduce the computation complexity, increase the speed of convolution and reduce the number of trainable parameters
2. [Albumentations](https://albumentations.ai) for image augmentations

## 2. Project structure
This project is organised as shown below,
```sh
.
├── Makefile
├── README.md
├── cifar10_playground.ipynb            # 
├── data                                #
│   ├── cifar-10-batches-py             #
│   │   ├── batches.meta                #
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── cifar-10-python.tar.gz
├── model_analysis.ipynb
├── models.py                           #
├── utils.py                            #
```

## 3. How to run 
1. Make sure `JupyterLab` is installed,
```sh
$ jupyter --version
Selected Jupyter core packages...
IPython          : 8.19.0
ipykernel        : 6.28.0
ipywidgets       : not installed
jupyter_client   : 8.6.0
jupyter_core     : 5.5.1
jupyter_server   : 2.12.1
jupyterlab       : 4.0.9
nbclient         : 0.9.0
nbconvert        : 7.13.1
nbformat         : 5.9.2
notebook         : not installed
qtconsole        : not installed
traitlets        : 5.14.0
```

If not, install it,
```sh
# Using pip:
$ pip install jupyterlab
# OR using Homebrew, a package manager for macOS and Linux
$ brew install jupyterlab
```

2. Clone this repository to your local machine.
```sh
$ git clone https://github.com/bensooraj/pytorch-s9-CIFAR10
$ cd pytorch-s9-CIFAR10
```

3. Start the lab!
```sh
$ make start-lab
```
This should automatically launch your default browser and open `http://localhost:8888/lab`.

All set!

## 4. Observations
### 4.1 Model summary
```sh
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Net                                      [1, 10]                   --
├─Sequential: 1-1                        [1, 32, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 32, 32, 32]           4,736
│    └─ReLU: 2-2                         [1, 32, 32, 32]           --
│    └─BatchNorm2d: 2-3                  [1, 32, 32, 32]           64
├─Sequential: 1-2                        [1, 64, 32, 32]           --
│    └─Conv2d: 2-4                       [1, 32, 32, 32]           832
│    └─Conv2d: 2-5                       [1, 64, 32, 32]           2,112
│    └─ReLU: 2-6                         [1, 64, 32, 32]           --
│    └─BatchNorm2d: 2-7                  [1, 64, 32, 32]           128
│    └─Dropout: 2-8                      [1, 64, 32, 32]           --
├─Sequential: 1-3                        [1, 64, 30, 30]           --
│    └─Conv2d: 2-9                       [1, 64, 30, 30]           36,928
│    └─ReLU: 2-10                        [1, 64, 30, 30]           --
│    └─BatchNorm2d: 2-11                 [1, 64, 30, 30]           128
│    └─Dropout: 2-12                     [1, 64, 30, 30]           --
├─Sequential: 1-4                        [1, 128, 15, 15]          --
│    └─Conv2d: 2-13                      [1, 128, 15, 15]          73,856
│    └─ReLU: 2-14                        [1, 128, 15, 15]          --
│    └─BatchNorm2d: 2-15                 [1, 128, 15, 15]          256
├─AvgPool2d: 1-5                         [1, 128, 1, 1]            --
├─Sequential: 1-6                        [1, 10]                   --
│    └─Linear: 2-16                      [1, 10]                   1,290
==========================================================================================
Total params: 120,330
Trainable params: 120,330
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 57.72
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 3.22
Params size (MB): 0.48
Estimated Total Size (MB): 3.71
==========================================================================================
```

### 4.2 Receptive field and accuracies
1. Traning accuracy: 57.23%
2. Testing accuracy: 62.44%
3. Receptive field: 45

## 5. Challenges
1. The MPS backend doesn't work properly with `shuffle=True` for [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#module-torch.utils.data).

## 6. Resources
1. [Depthwise-Separable convolutions in Pytorch](https://faun.pub/depthwise-separable-convolutions-in-pytorch-fd41a97327d0)
2. [PyTorch and Albumentations for image classification](https://albumentations.ai/docs/examples/pytorch_classification/)
