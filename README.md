# Pytorch-LapSRN
Implementation of paper [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](http://vllab1.ucmerced.edu/~wlai24/LapSRN/papers/cvpr17_LapSRN.pdf).
Refactoring and enhancement of the original code: https://github.com/BUPTLdy/Pytorch-LapSRN

![](http://vllab1.ucmerced.edu/~wlai24/LapSRN/images/network.jpg)

# Prerequisites

- Linux
- Python
- NVIDIA GPU >= 4GB
- pytorch
- torchvision

# Usage

- Train the model
```sh
python train.py --train_dir <path_photos>
```

- Test the model

```sh
python test.py
```

The work on this repo is not complete! Use on your own risk :)