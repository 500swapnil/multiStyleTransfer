from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.utils as vutils
import torch
import os
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in os.listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize
        self.prep = transforms.Compose([
                    transforms.Scale(fineSize),
                    transforms.ToTensor(),
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        contentName = self.image_list[index][:-4]
        styles = []
        for style in os.listdir(stylePath):
            if contentName in style:
                styleImg = default_loader(os.path.join(self.stylePath,style))
                styles.append(styleImg)

        contentImg = default_loader(contentImgPath)



        # resize
        styles_rsz = []
        for styleImg in styles:
            if(self.fineSize != 0):
                w,h = contentImg.size
                if(w > h):
                    if(w != self.fineSize):
                        neww = self.fineSize
                        newh = int(h*neww/w)
                        contentImg = contentImg.resize((neww,newh))
                        styleImg = styleImg.resize((neww,newh))
                else:
                    if(h != self.fineSize):
                        newh = self.fineSize
                        neww = int(w*newh/h)
                        contentImg = contentImg.resize((neww,newh))
                        styleImg = styleImg.resize((neww,newh))
            styles_rsz.append(styleImg)


        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        style_images = []
        for styleImg in styles_rsz:
            styleImg = transforms.ToTensor()(styleImg)
            styleImg.squeeze(0)
            style_images.append(styleImg)
        return contentImg.squeeze(0),style_images,self.image_list[index]

    def __len__(self):
        return len(self.image_list)