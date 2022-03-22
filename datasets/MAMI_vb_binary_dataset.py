import torch
import torch.utils.data as data
import torchvision.utils as utils
import torchvision
import torchvision.transforms as T

from PIL import Image
import numpy as np
import cv2
import os

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg


class MAMI_vb_binary_dataset(data.Dataset):

    def __init__(self, text, image_path, text_processor, label, max_length=128):
        self.text_processor = text_processor
        self.text = text
        self.label = label
        self.max_length = max_length
        self.image_path = image_path

    def __getitem__(self, index):
        return self.text_processor(self.text[index], padding="max_length", max_length=self.max_length, truncation=True,
                                   return_tensors='pt'), self.image_path[index], self.label[index]

    def load_image(self, filename):
        img = Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="float32")
        return data

    def __len__(self):
        return len(self.text)
