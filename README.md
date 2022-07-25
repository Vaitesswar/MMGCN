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
