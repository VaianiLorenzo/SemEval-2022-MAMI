# MAMI - Multimedia Automatic Misogyny Identification 
The proposed task, i.e. Multimedia Automatic Misogyny Identification ([MAMI](https://competitions.codalab.org/competitions/34175)), consists in the identification of misogynous memes, taking advantage of both text and images available as source of information. MAMI is part of [Sem-Eval 2022](https://semeval.github.io/SemEval2022/tasks).

This repository contains our solution for the subtask A of teh challange: a basic task about misogynous meme identification, where a meme should be categorized either as misogynous or not misogynous. 

## Methodology
Two models have been developed to address the proposed task. The former can be considered a baseline solutions that can be trained using text only, images only or both input modalities at the same time. Feature extractors are BERT and VGG16 for texts and images respectively. In case of multimodal approach, the modality embeddings are concatenated together after the extraction (late fusion). The classification is always performed by a MLP. Our second model is a combination of Mask R-CNN and VisualBERT. Mask R-CNN select the patches from the image to feed VisualBERT, together with textual tokens. Classification is performed again with MLP. There is also the possibility to exploit different pretraining of Mask R-CNN (COCO, LVIS or both) and different classification strategy (using CLS token or averaging all the output tokens).

## Repo Description
Folders:
- "datasets" folder: contains custom datasets classes, used to create dataloaders.
- "models" folder: contains our baseline model and Mask R-CNN + VisualBERT model.
- "utils" folder: contains useful snippets of code, including the training function, the collate functions needed by the dataloaders and the configurations of editable parameters to adjust the functioning of the other scripts.
 
Main files: 
- "main_dataloader_creation.py": allows to create train, validation and test dataloaders for both our models. 
- "main_statistics.py": it allows to compute some statistics about visual elements retrieved by Mask R-CNN in the dataset. 
- "main_test.py": allows to load a trained model to evaluate unlabeled samples.
- "main_training.py": calls the training function and allows to train both our models. 

## Citattion
````
@inproceedings{ravagli2022jrlv,
  title={JRLV at semeval-2022 task 5: The importance of visual elements for misogyny identification in memes},
  author={Ravagli, Jason and Vaiani, Lorenzo},
  booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  pages={610--617},
  year={2022}
}
