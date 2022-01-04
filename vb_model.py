import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from mlp import MLP

from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, VisualBertForPreTraining

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
import cv2

class MAMI_vb_binary_model(nn.Module):
    def __init__(self, vb_model_name='uclanlp/visualbert-nlvr2-coco-pre', modality="multimodal",
                 device=None, text_tokenizer=None
                 ):
        super().__init__()

        self.device = device
        self.model = VisualBertForPreTraining.from_pretrained(vb_model_name)  # this checkpoint has 1024 dimensional visual embeddings projection
        self.model.to(self.device)

    def forward(self, x_text, x_image):

        #cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        cfg_path = "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # ROI HEADS SCORE THRESHOLD
        #cfg['MODEL']['DEVICE'] = 'cpu' # if you are not using cuda
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model) # load weights
        checkpointer.load(cfg.MODEL.WEIGHTS)
        model.eval() # eval mode

        visual_attention_mask = []
        visual_token_type_ids = []
        visual_embeds = []

        for path in x_image:
            image = cv2.imread(path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            with torch.no_grad():
                images = model.preprocess_image(inputs)  # don't forget to preprocess
                features = model.backbone(images.tensor)  # set of cnn features
                proposals, _ = model.proposal_generator(images, features, None)  # RPN
                features_ = [features[f] for f in model.roi_heads.box_in_features]
                box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
                pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
                # output boxes, masks, scores, etc
                pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                # features of the proposed boxes
                feats = box_features[pred_inds]

                mask = [1] * len(feats)
                gap = 32 - len(feats)
                for _ in range(gap):
                    feats = torch.cat((feats, torch.stack([torch.tensor([0] * 1024).to(self.device)])), 0)
                    mask.append(0)

            visual_embeds.append(feats)
            visual_attention_mask.append(torch.tensor(mask).to(self.device))
            visual_token_type_ids.append(torch.stack([torch.tensor([0] * len(feats)).to(self.device)]))

        visual_embeds = torch.stack(visual_embeds)
        visual_attention_mask = torch.stack(visual_attention_mask)
        visual_token_type_ids = torch.stack(visual_token_type_ids)

        input_ids = [x['input_ids'][0] for x in x_text]
        attention_mask = [x['attention_mask'][0] for x in x_text]
        token_type_ids = [x['token_type_ids'][0] for x in x_text]
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        token_type_ids = torch.stack(token_type_ids).to(self.device)

        visual_embeds1 = visual_embeds
        visual_embeds2 = visual_embeds
        print(visual_embeds.shape)
        vb = torch.cat((visual_embeds1, visual_embeds2), 0)
        print(vb.shape)
        vb = torch.cat((visual_embeds1, visual_embeds2), 1)
        print(vb.shape)
        vb = torch.cat((visual_embeds1, visual_embeds2), 2)
        print(vb.shape)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                        visual_token_type_ids=visual_token_type_ids)

        return outputs

    def image_mean_pooling(self, model_output):
        result = torch.sum(model_output, 0) / model_output.size()[0]
        return result

    def text_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
