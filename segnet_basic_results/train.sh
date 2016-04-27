#!/bin/bash

#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --mail-user=yixi@cs.umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name="segnetflowtrain"
#SBATCH --gres=gpu

../caffe-segnet/build/tools/caffe train -gpu 0 -solver segnet_basic_solver.prototxt -weights basic_camvid_surg.caffemode

