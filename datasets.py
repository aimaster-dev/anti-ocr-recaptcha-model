# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
try:
    from .CONFIG import LettersInt
except:
    from CONFIG import LettersInt

content_range = LettersInt.content_range

def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert('RGB')

def make_dataset(data_path, content_range, range_len, pic_name_len):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        target_str = img_name.split('/')[-1].split('.')[0]
        assert len(target_str) == pic_name_len
        target = []
        for char in target_str:
            vec = [0] * range_len
            vec[content_range.find(char)] = 1
            target += vec
        samples.append((img_path, target))
    return samples

class CaptchaData(Dataset):

    def __init__(self, data_path, range_len=LettersInt.range_len, pic_name_len=LettersInt.PIC_NAME_LEN,
                 transform=None, target_transform=None, content_range=content_range):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.range_len = range_len
        self.pic_name_len = pic_name_len
        self.transform = transform
        self.target_transform = target_transform
        self.content_range = content_range
        self.samples = make_dataset(self.data_path, self.content_range,
                                    self.range_len, self.pic_name_len)


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, torch.Tensor(target)


class CaptchaDataOne(Dataset):

    def __init__(self, samples,transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
