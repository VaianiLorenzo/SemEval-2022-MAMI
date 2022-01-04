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

def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE'] = 'cpu'
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
    return cfg

def get_model(cfg):
    # build model
    model = build_model(cfg)
    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    # eval mode
    model.eval()
    return model

def prepare_image_inputs(cfg, img_list, model):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]
    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))
    batched_inputs = [{"image": convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in
                      img_list]
    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]
    # Convert to ImageList
    images = ImageList.from_tensors(images, model.backbone.size_divisibility)
    return images, batched_inputs

def get_features(model, images):
    features = model.backbone(images.tensor)
    return features

def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals

def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)
    box_features = box_features.reshape(1, 1000, 1024)  # depends on your config and batch size
    return box_features, features_list

def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals, shape):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    """
    outputs = FastRCNNOutputLayers(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    )
    """

    outputs = FastRCNNOutputLayers(shape, box2box_transform=box2box_transform, num_classes=2)

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes
    return boxes, scores, image_shapes

def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)
    return output_boxes

def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf

def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes

def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]


class MAMI_vb_binary_dataset(data.Dataset):

    def __init__(self, text, image_path, text_processor, label, max_length=128):
        self.text_processor = text_processor
        self.text = text
        self.label = label
        self.max_length = max_length
        self.image_path = image_path


    def __getitem__(self, index):
        '''
        img = self.load_image(self.image_path[index])
        cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = load_config_and_model_weights(cfg_path)
        model = get_model(cfg)
        images, batched_inputs = prepare_image_inputs(cfg, [img], model)
        features = get_features(model, images)
        proposals = get_proposals(model, images, features)
        box_features, features_list = get_box_features(model, features, proposals)
        pred_class_logits, pred_proposal_deltas = get_prediction_logits(model, features_list, proposals)
        boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals, box_features.shape)
        output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
        temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
        keep_boxes, max_conf = [], []
        for keep_box, mx_conf in temp:
            keep_boxes.append(keep_box)
            max_conf.append(mx_conf)
        MIN_BOXES = 10
        MAX_BOXES = 100
        keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
        visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
        '''

        return self.text_processor(self.text[index], padding="max_length", max_length=self.max_length, truncation=True, return_tensors='pt'), self.image_path[index], self.label[index]



    def load_image(self, filename):
        img = Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="float32")
        return data


    def __len__(self):
        return len(self.text)



