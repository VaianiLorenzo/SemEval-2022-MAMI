import torch
import torch.utils.data as data
import torchvision.utils as utils
import torchvision
import torchvision.transforms as T

from PIL import Image
import numpy as np


class MAMI_test_vb_binary_dataset(data.Dataset):

    def __init__(self, text, image_path, text_processor, max_length=128):
        self.text_processor = text_processor
        self.text = text
        self.max_length = max_length
        self.image_path = image_path

    def __getitem__(self, index):
        return self.text_processor(self.text[index], padding="max_length", max_length=self.max_length, truncation=True,
                                   return_tensors='pt'), self.image_path[index]

    def __len__(self):
        return len(self.text)
