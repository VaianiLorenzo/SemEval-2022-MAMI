import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from mlp import MLP

from transformers import AutoTokenizer, AutoModel


class MAMI_binary_model(nn.Module):
    def __init__(self, text_model_name="bert-base-cased",
                 device=None, text_tokenizer=None
                 ):
        super().__init__()

        self.device = device

        # instantiate sBERT, with sBERT tokenizer
        self.text_model = AutoModel.from_pretrained(text_model_name)
        if text_tokenizer == None:
            text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model.resize_token_embeddings(len(text_tokenizer))
        self.text_model = self.text_model.to(self.device)
        self.len_text_embeddings = self.text_model.config.hidden_size

        self.intermediate_embeddings_len = self.len_text_embeddings + 2048

        # instantiate ResNet for image  processing
        self.image_model = models.resnet50(pretrained=True)
        self.avg_layer = self.image_model._modules.get('avgpool')

        # instantiate MLP
        self.mlp = MLP(input_dim=self.intermediate_embeddings_len, output_dim=1)
        self.mlp = self.mlp.to(self.device)

    def forward(self, x_text, x_image):

        # image processing with resnet
        # get features extracted after last avg pool layer, before fully connected part
        image_embeddings = []
        for element in x_image:
            image_embedding = torch.zeros(2048)

            def copy_data(m, i, o):
                image_embedding.copy_(o.data.reshape(o.data.size(1)))

            h = self.avg_layer.register_forward_hook(copy_data)
            self.image_model(torch.stack([element]).to(self.device))
            h.remove()

            image_embeddings.append(image_embedding)
        image_embeddings = torch.stack(image_embeddings).to(self.device)

        # text processing with BERT
        # compute the average between all output tokens of the BERT encoder
        input_ids = [x['input_ids'][0] for x in x_text]
        attention_mask = [x['attention_mask'][0] for x in x_text]
        input_ids = torch.stack(input_ids)

        model_output = self.text_model(input_ids.to(self.device))
        sentence_embeddings = self.text_mean_pooling(model_output, torch.stack(attention_mask).to(self.device))
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # concat
        # obtain 768 + 512 = 1280
        overall_embedding = torch.cat((image_embeddings, sentence_embeddings), 1)

        # pass to MLP ()
        # return MLP.forward(...)
        return torch.flatten(self.mlp(overall_embedding))

    def image_mean_pooling(self, model_output):
        result = torch.sum(model_output, 0) / model_output.size()[0]
        return result

    def text_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
