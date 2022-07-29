# Multi-Range Mixed Graph Convolution Network for Skeleton-Based Action Recognition (MMGCN) #

## Introduction ##

Skeleton-based action recognition is a long-standing task in computer vision which aims to distinguish different human
actions by identifying their unique characteristic patterns in the input data. Most of the existing GCN-based models
developed for this task primarily model the skeleton graph as either directed or undirected. Furthermore, these models also
restrict the receptive field in the temporal domain to a fixed range which significantly inhibits their expressibility. Therefore,
we propose a mixed graph network comprising both directed and undirected graph networks with a multi-range temporal
module called MMGCN. In this way, the model can benefit from the different interpretations of the same action by the
different graphs. Adding on, the multi-range temporal module enhances the model’s expressibility as it can choose the
appropriate receptive field for each layer, thus allowing the model to dynamically adapt to the input data. With this
lightweight MMGCN model, we further show that deep learning models can learn the underlying patterns in the data and
model large receptive fields without additional semantics or high model complexity. Finally, this model achieved state-ofthe-art results on three benchmark datasets namely NTU RGB+D, NTU RGB+D 120 and Northwestern-UCLA despite its
low model complexity thus proving its effectiveness.

## Architecture ##
![Fig 2](https://user-images.githubusercontent.com/81757215/180739256-bbc367d2-c2f4-4cc2-9d60-5c39a0f8ce85.jpg)
Description: The general architecture of each stream in the four-stream MMGCN model. The three numbers
beside each block represent the number of input channels, the number of output channels and the stride,
respectively in each convolution layer. GAP, BN and FC represent the global average pooling, batch
normalization and the fully connected layer respectively. The details of the spatio-temporal (ST) block are
illustrated at the bottom. τ and 𝑑 denote the kernel size and dilation for the temporal dimension.

## Prerequisites ##
The code is built with the following libraries:

- Python 3.6
- Anaconda
- PyTorch 1.3
- EasyDict

## Data Preparation ##
The  json skeleton data for NW+UCLA dataset was obtained from the following link and was provided in CTR-GCN's Github page.
https://github.com/Uason-Chen/CTR-GCN

The skeleton data for NTU RGB+D and NTU RGB+D 120 datasets was obtained by extracting the 3D coordinates from raw .skeleton files obtained
from NTU ROSE lab.
https://rose1.ntu.edu.sg/dataset/actionRecognition/

Process the data using the appropriate Python files in data_gen folder for the different datasets. Note that the motion and bone data is preprocessed for NTU datasets. On the other hand, bone and motion data are processed during runtime for NW-UCLA dataset as seen in feeder_ucla.py in feeders folder.

## Models ##
The GCN models can be found under "models" folder for each dataset.

## Training ##
Model architecture and feeder have to be modified in lines 25 and 27 of training.py file in training folder respectively. Note that the feeder file varies for NTU and NW-UCLA datasets. The parameters of the optimizer, training schedule and path to input data can be changed in main.py file. "processor" class will take in the parameters and will start training the model upon initiating "processor.start" command in main.py file.

## Testing ##
In main.py, update the "phase" attribute from "train" to "test" and specify the path to weights of the pretrained model in "weights" attribute.

## Weights and Test Results ##
The individual model weights and the predicted action labels for the test samples can be found in the "model weights and results" folder.

## Citing paper ##
Please refer to the following link for the full details of this model.
- https://dr.ntu.edu.sg/handle/10356/156866

If you find the repository or paper useful, please cite as follows.
- U S Vaitesswar (2022). Skeleton-based human action recognition with graph neural networks. Master's thesis, Nanyang Technological University, Singapore. https://hdl.handle.net/10356/156866
