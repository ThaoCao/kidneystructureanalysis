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

rootdir = '/project/mclark/For/Thao/GAN/Lupus_Nephritis_60ch/'
savedir = '/project/mclark/For/Thao/GAN/Lupus_Nephritis_60ch_ds/'

os.makedirs(savedir, exist_ok=True)

# Iterate through samples
for sample in os.listdir(rootdir):
    sample_path = os.path.join(rootdir, sample)
    
    # Skip if not a file or not a .tif file
    if not os.path.isfile(sample_path) or not sample.endswith('.tif'):
        continue
    
    print(f"Processing {sample}...")
    
    img = imread(sample_path)
    r, c = np.shape(img)
    img_resized = resize(img, (r//10, c//10), order=3, anti_aliasing=False, preserve_range=True)
    
    # Save resized image with the same name in savedir
    output_path = os.path.join(savedir, sample)
    imwrite(output_path, img_resized.astype(np.uint16))
    print(f"Saved resized image to {output_path}")

