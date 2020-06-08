# SGRN_torch
Implementation of [Spatial-aware Graph Relation Network for Large-scale Object Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Spatial-Aware_Graph_Relation_Network_for_Large-Scale_Object_Detection_CVPR_2019_paper.pdf)

This code is modified based on [ruotianluo](https://github.com/ruotianluo/pytorch-faster-rcnn) Faster R-CNN code.

I implement code only for Faster R-CNN / Visual Genome Dataset / Resnet101 backbone.

------------------------------------------------------------------------

## Prepare Dataset & Pre-weight
- Download Annotation File [link](https://drive.google.com/open?id=1l1MDX5xrXYzLq8zsjn-NSW-JmSCvWUy1)
-->  Unzip zip file to $your_data_path/vg
- Download Visual Genome Image Files [zip1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [zip2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
-->  Unzip 2 zip files to $your_data_path/vg/JPEGImages/*.jpg
- Download Imagenet pre-trained weight for resnet 101 [link](https://drive.google.com/file/d/1HXXVHADWy3mjegvtTtzpmdj1IdvnPL16/view?usp=sharing)
-->  Copy file to $your_preweight_path/
- Download trained weight of Faster RCNN [link](https://drive.google.com/file/d/11k9WAGB5YmYwKDUu7kLW5E5q22AnOhzW/view?usp=sharing)
-->  Copy file to $your_weight_path/FRCNN/ and use it for training SGRN module
------------------------------------------------------------------------

## Requirements
 - torch > 1.4.0
 - torchvision > 0.5.0
 - [torch-geometric](https://github.com/rusty1s/pytorch_geometric)

------------------------------------------------------------------------

## Train
 - Train Faster RCNN
```Shell
python tools/trainval_net.py --cfg "experiments/cfgs/res101.yml" --tag "FRCNN" --net "res101" --weight $your_preweight_path/res101.pth --iters 2400000
```
- Train SGRN
```Shell
python tools/trainval_net.py --cfg "experiments/cfgs/res101_gcn.yml" --tag "SGRN" --net "SGRN" --weight "your_weight_path/FRCNN/res101_faster_rcnn_iter_1200000.pth" --iters 1200000
```
------------------------------------------------------------------------
## Test
 - Test Faster RCNN
```Shell
python tools/test_net.py --cfg "experiments/cfgs/res101.yml" --model "your_weight_path/FRCNN/res101_faster_rcnn_iter_1200000.pth"  --net "res101"
```

- Download trained weight of SGRN [link](https://drive.google.com/file/d/1O2rGJqRXVlW9LVhwN5IvqDOLqWmYdPpP/view?usp=sharing)
-->  Copy file to $your_weight_path/SGRN/

- Test SGRN
```Shell
python tools/test_net.py --cfg "experiments/cfgs/res101_gcn.yml" --model "your_weight_path/SGRN/res101_faster_rcnn_iter_1200000.pth"  --net "SGRN"
```

------------------------------------------------------------------------
## Result
Faster RCNN Mean Average Precision : 11.2
SGRN Mean Average Precision : 11.5
