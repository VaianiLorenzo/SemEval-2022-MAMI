import cv2
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from transformers import VisualBertModel

from mlp import MLP


class MAMI_vb_binary_model_ft(nn.Module):

    def __init__(self, path_tl_model, device, n_frozen_layers=6):
        super().__init__()

        self.device = device

        # Load model from transfer learning phase
        self.model = torch.load(path_tl_model)

        self.maskr_modality = self.model.maskr_modality
        self.class_modality = self.model.class_modality

        # Set evaluation mode for Mask-R layers
        if self.maskr_modality == "coco" or self.maskr_modality == "both":
            self.model.maskr_coco.eval()
        if self.maskr_modality == "lvis" or self.maskr_modality == "both":
            self.model.maskr_lvis.eval()

        # Freeze the first n_frozen_layers of VisualBERT
        vb_modules = [self.model.visual_bert.embeddings, *self.model.visual_bert.encoder.layer[:n_frozen_layers]]
        for module in vb_modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x_text, x_image):
        visual_embeds = None

        if self.maskr_modality == "coco" or self.maskr_modality == "both":
            self.model.maskr_coco.eval()
            feats_coco = self.calculate_feats_patches(self.model.maskr_coco, x_image)

            visual_embeds = feats_coco

        if self.maskr_modality == "lvis" or self.maskr_modality == "both":
            self.model.maskr_lvis.eval()
            feats_lvis = self.calculate_feats_patches(self.model.maskr_lvis, x_image)

            if visual_embeds is None:
                visual_embeds = feats_lvis
            else:
                for i in range(len(visual_embeds)):
                    visual_embeds[i] = torch.cat((visual_embeds[i], feats_lvis[i]), 0)

        visual_attention_mask = []
        visual_token_type_ids = []

        for i in range(len(visual_embeds)):
            # Cut out patch features if patches are more than 32
            if len(visual_embeds[i]) > 32:
                visual_embeds[i] = visual_embeds[i][:32]

            mask = [1] * len(visual_embeds[i])
            gap = 32 - len(visual_embeds[i])
            for _ in range(gap):
                visual_embeds[i] = torch.cat((visual_embeds[i], torch.stack([torch.tensor([0] * 1024).to(self.device)])), 0)
                mask.append(0)

            visual_attention_mask.append(torch.tensor(mask).to(self.device))
            visual_token_type_ids.append(torch.tensor([1] * len(visual_embeds[i])).to(self.device))

        visual_embeds = torch.stack(visual_embeds)
        visual_attention_mask = torch.stack(visual_attention_mask)
        visual_token_type_ids = torch.stack(visual_token_type_ids)

        input_ids = [x['input_ids'][0] for x in x_text]
        attention_mask = [x['attention_mask'][0] for x in x_text]
        token_type_ids = [x['token_type_ids'][0] for x in x_text]
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        token_type_ids = torch.stack(token_type_ids).to(self.device)

        outputs = self.model.visual_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                   visual_token_type_ids=visual_token_type_ids)
        outputs_embeddings = outputs.last_hidden_state

        if self.class_modality == "cls":
            cls_out_embeddings = outputs_embeddings[:, 0]
            predictions = torch.flatten(self.model.mlp(cls_out_embeddings))
        else:
            l = []
            for i in range(len(outputs_embeddings)):
                average = self.global_average_pooling(outputs_embeddings[i])
                l.append(average)
            out_embedding_avg = torch.stack(l)

            predictions = torch.flatten(self.model.mlp(out_embedding_avg))

        return predictions

    def calculate_feats_patches(self, model, x_image):
        visual_embeds = []

        inputs = []
        for path in x_image:
            image = cv2.imread(path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})

        with torch.no_grad():
            images = model.preprocess_image(inputs)  # don't forget to preprocess
            features = model.backbone(images.tensor)  # set of cnn features
            proposals, _ = model.proposal_generator(images, features, None)  # RPN

            for i in range(len(proposals)):
                # features_ = [torch.stack([features[f][i]]) for f in model.roi_heads.box_in_features]
                features_single = {}
                features_ = []
                for f in model.roi_heads.box_in_features:
                    tensor = torch.stack([features[f][i]])
                    features_.append(tensor)
                    features_single[f] = tensor

                box_features = model.roi_heads.box_pooler(features_, [proposals[i].proposal_boxes])
                box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, [proposals[i]])
                pred_instances = model.roi_heads.forward_with_given_boxes(features_single, pred_instances)
                # output boxes, masks, scores, etc
                pred_instances = model._postprocess(pred_instances, inputs,
                                                    images.image_sizes)  # scale box to orig size
                # features of the proposed boxes
                feats = box_features[pred_inds]
                visual_embeds.append(feats)

        return visual_embeds

    def image_mean_pooling(self, model_output):
        result = torch.sum(model_output, 0) / model_output.size()[0]
        return result

    def text_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def global_average_pooling(self, model_output):
        result = torch.sum(model_output, 0) / model_output.size()[0]
        return result
