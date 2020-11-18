import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from dataLoader import Dataset
from wct import *
from torch.utils.serialization import load_lua
import time

parser = argparse.ArgumentParser()

alpha = 1
CONTENT_DIR = 'input/content'
STYLE_DIR = 'input/style'
OUTPUT_DIR = 'output/'
out_size = 512
batch_size = 1
cuda = True
workers = 2


parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7')

args = parser.parse_args()

try:
    os.makedirs(OUTPUT_DIR)
except OSError:
    pass

# Data loading code
dataset = Dataset(CONTENT_DIR, STYLE_DIR,out_size)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

wct = WCT(args)
def styleTransfer(contentImg,styleImg,imname,csF):

    sF5 = wct.e5(styleImg)
    cF5 = wct.e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5,sF5,csF,alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(styleImg)
    cF4 = wct.e4(Im5)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4,sF4,csF,alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(styleImg)
    cF3 = wct.e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3,sF3,csF,alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(styleImg)
    cF2 = wct.e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2,sF2,csF,alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(styleImg)
    cF1 = wct.e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1,sF1,csF,alpha)
    Im1 = wct.d1(csF1)

    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(Im1.data.cpu().float(),os.path.join(OUTPUT_DIR,imname))
    return

cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
csF = Variable(csF)
if(cuda):
    cImg = cImg.cuda(0)
    sImg = sImg.cuda(0)
    csF = csF.cuda(0)
    wct.cuda(0)
for i,(contentImg,styleImg,imname) in enumerate(loader):
    imname = imname[0]
    print('Transferring Style to ' + imname)
    if (cuda):
        contentImg = contentImg.cuda(0)
        styleImg = styleImg.cuda(0)
    cImg = Variable(contentImg,volatile=True)
    sImg = Variable(styleImg,volatile=True)
    start_time = time.time()
    # WCT Style Transfer
    styleTransfer(cImg,sImg,imname,csF)
    end_time = time.time()
    print('Time Elapsed: %f\n\n' % (end_time - start_time))
    