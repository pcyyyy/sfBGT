# sfBGT

This repository is for the paper "Joint Structural-Functional Brain Graph Transformer".

# Overview

* `main_sfbgt.py` runs the sfBGT training pipeline.
* `nets/sfbgt_graph_classification` contains the sfBGT model used for multimodal brain graph classification.
* `train/train_sfbgt.py` contains training and evaluation code for multi-class brain disease prediction.
* `data/sfbgt_dataset.py` loads the preprocessed subject-level ROI features and pairwise multimodal connectivities.
* `configs/SFBGT_CUSTOM.json` stores the default sfBGT experiment configuration.
* `layers` contains reusable neural network building blocks shared by the model.
