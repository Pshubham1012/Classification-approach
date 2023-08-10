# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:53:05 2023

@author: shubham_p
"""

import glob
import shutil
import os
import numpy as np
src_dir_im = r'E:\JHU_DATA _FINAL\data\Shanghaitech\part_A\validation\images'
src_dir_gt = r'E:\JHU_DATA _FINAL\data\Shanghaitech\part_A\validation\ground-truth'
dst_dir_low = r'E:\JHU_DATA _FINAL\data\Shanghaitech\part_A\val_low'
dst_dir_high = r'E:\JHU_DATA _FINAL\data\Shanghaitech\part_A\val_high'
dst_dir_med =  r'E:\JHU_DATA _FINAL\data\Shanghaitech\part_A\val_med'
# =============================================================================
# imnames = [f for f in os.listdir(src_dir_im) if f.endswith('.jpg')]
# gtnames =  [f for f in os.listdir(src_dir_gt) if f.endswith('.txt')]
# shutil.move(os.path.join(src_dir_im,file), dst_dir_img)
# =============================================================================
#gd_path = 'E:\\JHU_DATA _FINAL\\data\Shanghaitech\part_A\\train_data\\ground-truth\\GT_IMG_8.txt'
#keypoints_csv = np.loadtxt(gd_path, delimiter=',')
#print(len(keypoints_csv))
for jpgfile,txtfile in zip(glob.iglob(os.path.join(src_dir_im, "*.jpg")),glob.iglob(os.path.join(src_dir_gt, "*.txt"))):
    kp = np.loadtxt(txtfile,delimiter=',')
    n = len(kp)
    if n<=50:
        shutil.copy(jpgfile, os.path.join(dst_dir_low,'images'))
        shutil.copy(txtfile, os.path.join(dst_dir_low,'ground-truth'))
    elif n>50 and n<=500:
        shutil.copy(jpgfile, os.path.join(dst_dir_med,'images'))
        shutil.copy(txtfile, os.path.join(dst_dir_med,'ground-truth'))
    elif n>500:
        shutil.copy(jpgfile, os.path.join(dst_dir_high,'images'))
        shutil.copy(txtfile, os.path.join(dst_dir_high,'ground-truth'))
    #shutil.move(jpgfile, dst_dir_img)
#for txtfile in glob.iglob(os.path.join(src_dir, "*.txt")):
    #shutil.move(txtfile, dst_dir_gt)


# =============================================================================
# i = 0
# for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
#     x = mats[i]['image_info']
#     count = x[0][0][0][0][0].shape[0]
#     dst_dir = src_dir
#     if(count <= 400):
#         dst_dir = dst_dir_low
#     else:
#         dst_dir = dst_dir_high
#     i = i + 1
#     shutil.copy(jpgfile, dst_dir)
# =============================================================================
