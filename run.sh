#!/bin/bash
#/iag_ad_01/ad/yuanweizhong/miniconda/vggt
export CUDA_VISIBLE_DEVICES=1
# time python moge/scripts/infer.py -i /iag_ad_01/ad/yuanweizhong/huzeyu/vggt/scene/images/frame0001.png -o save_opencv_fov30_v1 --maps --ply --fov_x 30 --version v1
# time python moge/scripts/infer.py -i /iag_ad_01/ad/yuanweizhong/huzeyu/vggt/scene/images/ -o save_opencv_fov30_v2_normal --maps --ply --fov_x 30 --version v2
time python moge/scripts/infer.py -i /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data/images -o /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data/moge_images --maps --ply --fov_x 30 --version v2
