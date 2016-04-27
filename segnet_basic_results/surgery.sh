#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -N 1
#SBATCH --gres=gpu
#SBATCH --partition=scavenger



python ../../face_segmentation_finetune/utils/surgery_flow.py \
-f '../Models/segnet_basic_inference.prototxt' \
-c '../Models/Inference/segnet_basic_camvid.caffemodel' \
-t 'segnet_basic_deploy.prototxt' \
-o 'basic_camvid_surg.caffemodel' \
--fromlayer='conv1' \
--tolayer='conv1_flow'

