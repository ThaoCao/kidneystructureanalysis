#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:01:03 2024

@author: thaocao
"""
import os
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize
from tqdm import tqdm

rootdir = '/project/mclark/SCAMPI_datasets/'
datasets = ['Mixed_Rejection_60ch']
fullsize = 'Normalized_composites'
maskdir = 'Glomeruli_manual_GT_mask_upsized_temp'
sdir = 'Glomeruli_manual_GT_mask_upsized'

# Iterate through datasets
for dataset in tqdm(datasets, desc="Datasets"):
    maskdir_path = os.path.join(rootdir, dataset, maskdir)
    fullsize_dir = os.path.join(rootdir, dataset, fullsize)
    output_mask_dir = os.path.join(rootdir, dataset, sdir)
    os.makedirs(output_mask_dir, exist_ok=True)
    

    # Iterate through images in the mask directory
    images = os.listdir(maskdir_path)
    for image in tqdm(images, desc=f"Processing {dataset}", leave=False):
        image_path = os.path.join(maskdir_path, image)
        sample = os.path.splitext(image)[0].split('_')[0]
        area = os.path.splitext(image)[0].split('_')[1]
        area = area.replace('.tif','')

        # Find the matched fullsize image in the fullsize_dir by using sample and area
        matched_fullsize_dir = os.path.join(fullsize_dir, sample, area)
        matched_fullsize_img = os.listdir(matched_fullsize_dir)[0] if os.path.exists(matched_fullsize_dir) else print(f"No matching fullsize directory found for {sample}_{area}")
        fullsize_img = imread(os.path.join(matched_fullsize_dir, matched_fullsize_img))
        r, c = np.shape(fullsize_img)

        # Read and resize the sample image
        img = imread(image_path)
        img_resized = resize(img, (r, c), order=0, anti_aliasing=False, preserve_range=True)

        # Save the resized mask
        output_filename = f"{sample}_{area}.tif"
        output_path = os.path.join(output_mask_dir, output_filename)
        imwrite(output_path, img_resized.astype(np.dtype(img.dtype)))

        print(f"Processed and saved: {output_path}")


        