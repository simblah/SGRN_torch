#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Reval = re-eval. Re-evaluate saved detections.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
# from model.test import apply_nms
from torchvision.ops import nms
from model.config import cfg
from datasets.factory import get_imdb
import pickle
import os, sys, argparse
import numpy as np


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument(
        'output_dir', nargs=1, help='results directory', type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to re-evaluate',
        default='voc_2007_test',
        type=str)
    parser.add_argument(
        '--matlab',
        dest='matlab_eval',
        help='use matlab for evaluation',
        action='store_true')
    parser.add_argument(
        '--comp',
        dest='comp_mode',
        help='competition mode',
        action='store_true')
    parser.add_argument(
        '--nms', dest='apply_nms', help='apply nms', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def from_dets(imdb_name, output_dir,): # args):
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(False) #args.comp_mode)
    imdb.config['matlab_eval'] = False #args.matlab_eval
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = pickle.load(f)

    # print('Applying NMS to all detections')
    # nms_dets = nms(dets, cfg.TEST.NMS)
    # print("dets = ", dets)
    nms_dets = dets

    # if args.apply_nms:
    #     print('Applying NMS to all detections')
    #     nms_dets = nms(dets, cfg.TEST.NMS)
    # else:
    #     nms_dets = dets

    print('Evaluating detections')
    imdb.evaluate_detections(nms_dets, output_dir)


if __name__ == '__main__':
    # args = parse_args()

    output_dir = "D:/hoseok/project/pytorch-faster-rcnn/output/res101/voc_2007_test/default/res101_faster_rcnn_iter_70000/" #os.path.abspath(args.output_dir[0])
    imdb_name = 'visual_genome_test' #'voc_2007_test'
    from_dets(imdb_name, output_dir) #, args)
