# Pytorch-LapSRN
Implementation of paper [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](http://vllab1.ucmerced.edu/~wlai24/LapSRN/papers/cvpr17_LapSRN.pdf)
\+ Perceptual loss instead of MSE.

Perseptual loss: VGG_16 with one input channel (Y-channel) with random weights. It's suitable according to [A Powerful Generative Model Using Random Weights
for the Deep Image Representation](https://papers.nips.cc/paper/6568-a-powerful-generative-model-using-random-weights-for-the-deep-image-representation.pdf)

Based on original code: https://github.com/BUPTLdy/Pytorch-LapSRN (a lot of refactoring and enhancements have been made).

![](http://vllab1.ucmerced.edu/~wlai24/LapSRN/images/network.jpg)

# Prerequisites

- Linux (Ubuntu preferably)
- Python
- NVIDIA GPU
- pytorch, torchvision

# Usage

- Train the model
```sh
python train.py --train_dir <path_photos>
```

- Test the model

```sh
python test.py
```

PS. Work on this repo is not complete (a bit of hardcode).