#-*- coding: utf8 -*-
import sys, os, copy
import torch
from torch import nn
from vgg import vgg16_bn
from __init__ import IVGG_PATH


class ContentLoss(nn.Module):

    def __init__(self, weight):
        # TODO: smth with weight: move to perceptual
        super(ContentLoss, self).__init__()
        self.weight = weight
        self.output = None

    def forward(self, input):
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        raise ValueError, 'Unable run backward while being inactive'


class PerceptualLoss(nn.Module):
    def __init__(self, cnn, content_losses):
        super(PerceptualLoss, self).__init__()
        self.cnn = cnn
        if not content_losses:
            raise ValueError, 'No content losses given'
        self.content_losses = content_losses
        self.criterion = torch.nn.MSELoss()
        self.loss = None

    def forward(self, input, target):
        outputs, targets = [], []
        self.cnn(input)
        for content_loss in self.content_losses:
            outputs.append(content_loss.output.clone())

        self.cnn(target)
        for content_loss in self.content_losses:
            targets.append(content_loss.output.clone().detach())

        loss = 0.
        for i in xrange(len(self.content_losses)):
            loss += self.criterion(outputs[i], targets[i])

        self.loss = loss
        return self.loss

    def backward(self, retain_variables=True):
        return self.loss.backward(retain_variables=retain_variables)


def create_discriptor_net(content_weight=1.0, layers=None, pretrained=False):
    "Weights are not used"
    if layers is None:
        layers = ['relu_10'] #eq. relu4_2
    #VGG-random
    if not pretrained:
        cnn = vgg16_bn()
    else:
        cnn = torch.load(IVGG_PATH)

    cnn = cnn.features.cuda()
    content_losses = []

    #copy VGG into a new model with loss layers
    model = nn.Sequential().cuda()

    i = 1
    xlist = isinstance(cnn, torch.nn.DataParallel) and cnn.module or cnn
    for j, layer in enumerate(list(xlist)):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in layers:
                content_loss = ContentLoss(content_weight).cuda()
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append( content_loss )

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in layers:
                content_loss = ContentLoss(content_weight).cuda()
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)

    #cnn = cnn.cpu()
    del cnn
    model.eval()

    return model, content_losses


def test():
    model, loss = create_discriptor_net(layers=['relu_4', 'relu_6', 'relu_8', 'relu_10'], pretrained=True)
    from __init__ import TEST_DIR, DATASET_DIR
    from PIL import Image
    from torchvision.transforms import ToTensor
    from torch.autograd import Variable

    def load_image(path):
        img = Image.open(path).convert('YCbCr').split()[0]
        img = img.resize((320, 480), Image.BICUBIC)
        input = (ToTensor()(img)).unsqueeze(0)
        return input

    image_path = os.path.join(TEST_DIR, '14092_div2.jpg')
    input = load_image(image_path)
    input = Variable(input.cuda(), requires_grad=False)

    target_path = os.path.join(DATASET_DIR, '14092.jpg')
    target = load_image(target_path)
    target = Variable(target.cuda(), requires_grad=False)
    print 'loss', loss
    criterion = PerceptualLoss(model, loss).cuda()
    x = criterion(input, target)
    print x.backward()
    print x


if __name__ == '__main__':
    sys.exit(test())
