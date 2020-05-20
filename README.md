# SGRN_torch
Implementation of [Spatial-aware Graph Relation Network for Large-scale Object Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Spatial-Aware_Graph_Relation_Network_for_Large-Scale_Object_Detection_CVPR_2019_paper.pdf)

This code is modified based on [ruotianluo](https://github.com/ruotianluo/pytorch-faster-rcnn) Faster R-CNN code.

I implement code only for Faster R-CNN / Visual Genome Dataset / Resnet101 backbone.

1. Prepare Dataset
- Download Annotation File [link](https://drive.google.com/open?id=1l1MDX5xrXYzLq8zsjn-NSW-JmSCvWUy1)
- Unzip zip file to $your_data_path\vg
- Download Visual Genome Image Files [zip1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [zip2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- Unzip 2 zip files to $your_data_path\vg\JPEGImages\*.jpg


2. Requirements
 torch > 1.4.0
 torchvision > 0.5.0
 [torch-geometric](https://github.com/rusty1s/pytorch_geometric)
