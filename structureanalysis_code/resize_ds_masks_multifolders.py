#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:19:36 2024
@author: thaocao
Change order of the resize (0 is most rigged, 3 is most smooth)
"""
import os
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize

rootdir = '/project/mclark/For/Thao/added/'
folders = ['artery', 'glomeruli']
savedir = '/project/mclark/For/Thao/added/ds10/'

# Iterate through each folder
for folder in folders:
    folder_path = os.path.join(rootdir, folder)
    save_folder_path = os.path.join(savedir, folder)
    os.makedirs(save_folder_path, exist_ok=True)

    print(f"\nProcessing folder: {folder}")

    for sample in os.listdir(folder_path):
        sample_path = os.path.join(folder_path, sample)

        # Skip if not a file or not a .tif file
        if not os.path.isfile(sample_path) or not sample.endswith('.tif'):
            continue

        print(f"  Processing {sample}...")

        img = imread(sample_path)
        r, c = np.shape(img)
        img_resized = resize(img, (r//10, c//10), order=0, anti_aliasing=False, preserve_range=True)

        # Save resized image with the same name in save_folder_path
        output_path = os.path.join(save_folder_path, sample)
        imwrite(output_path, img_resized.astype(np.uint16))
        print(f"  Saved resized image to {output_path}")
