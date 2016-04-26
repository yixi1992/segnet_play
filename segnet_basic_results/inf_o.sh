#!/bin/bash

#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="segnetbasicinfo"
#SBATCH --partition="scavenger"


python ../Scripts/test_segmentation_camvid.py --model ../Models/segnet_basic_inference.prototxt --weights ../Models/Inference/segnet_basic_camvid.caffemodel --iter 233

