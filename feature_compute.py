import torch
import math
import json
import time
import os
import torch.utils.data
import numpy as np
from torch import nn, optim
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class Flatten(nn.Module):
    def __init__(self, full:bool=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        vgg = VGG(make_layers(cfg['E'], batch_norm=True))
        vgg.load_state_dict(torch.load('../vgg19_bn-6002323d.pth'))	
        ls = [l for l in vgg.features]+ [nn.AdaptiveMaxPool2d(1), Flatten()]
        #ls = [l for l in vgg.features]
        self.features = nn.Sequential(*ls)

#Some transformations
#scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

device = torch.device('cuda')
#Extracting features of an image

feac = None
fea = Features()
fea = fea.to(device)
reslist = []
count = 0 #exist

dict = {}
start = time.time()

#Example of a line from exist.txt
#bddreduce100k/images/train/fe172415-3c36f3d1.jpg
with open('exist.txt','r') as fp:
    for line in fp:
        pathimg = line
        imn = line.split('/')[3].strip()
        dict[imn]=count
        image = '../../../'+pathimg.strip()
        im = Image.open(image)
        t_img = Variable(normalize(to_tensor(im)).unsqueeze(0))
        t_img = t_img.to(device)
        res = fea.features(t_img)
        resl = res.tolist()
        reslist.append(resl)
        count=count+1
        print(count)

resar = np.array(reslist)
print(resar.shape) #features of the images present in exist.txt
resar = np.squeeze(resar,1)
np.save('res_train_exist_10k.npy',resar)
#Similarly find the features for all the sets
#When you do so, change the array name to say res_train_set1_30k.npy or res_train_set2_30k.npy (helpful to be referred later)
print(time.time() - start)

with open('dict_index_exist_10k.json','w') as fp:
	json.dump(dict,fp)

#dict['00067cfb-5adfaaa7.jpg']=1 numbering the image for post usage