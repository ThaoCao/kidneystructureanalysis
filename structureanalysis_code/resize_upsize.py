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
datasets = ['Duke_TCMR_60ch','Tcell_Mediated_Rejection_60ch','Mixed_Rejection_60ch']
fullsize = 'Normalized_composites'
tissue_types = ['glomeruli']
maskdir = 'masks'
rgbdir = 'glomeruli'
sdir = 'masks_fullsize'
rgbsdir = 'glomeruli_input_fullsize'

# Iterate through datasets
for dataset in tqdm(datasets, desc="Datasets"):
    for tissue_type in tissue_types:
        maskdir_path = os.path.join(rootdir, dataset, tissue_type, maskdir)
        rgbdir_path = os.path.join(rootdir, dataset, rgbdir)
        fullsize_dir = os.path.join(rootdir, dataset, fullsize)
        output_mask_dir = os.path.join(rootdir, dataset, tissue_type, sdir)
        os.makedirs(output_mask_dir, exist_ok=True)
        output_rgb_dir = os.path.join(rootdir, dataset, tissue_type, rgbsdir)
        os.makedirs(output_rgb_dir, exist_ok=True)

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

        # Iterate through rgb images in the rgb directory
        rgb_images = [f for f in os.listdir(rgbdir_path) if f.endswith('.tif')]
        for rgb_image in tqdm(rgb_images, desc=f"Processing RGB {dataset}", leave=False):
            rgb_image_path = os.path.join(rgbdir_path, rgb_image)
            sample = os.path.splitext(rgb_image)[0].split('_')[0]
            area = os.path.splitext(rgb_image)[0].split('_')[1]
            area = area.replace('.tif','')

            # Find the matched fullsize image in the fullsize_dir by using sample and area
            matched_fullsize_dir = os.path.join(fullsize_dir, sample, area)
            matched_fullsize_img = os.listdir(matched_fullsize_dir)[0] if os.path.exists(matched_fullsize_dir) else print(f"No matching fullsize directory found for {sample}_{area}")
            fullsize_img = imread(os.path.join(matched_fullsize_dir, matched_fullsize_img))
            r, c = np.shape(fullsize_img)

            # Read and resize the rgb image
            rgb_img = imread(rgb_image_path)

            rgb_img_resized = resize(rgb_img, (r, c, 3), order=0, anti_aliasing=False, preserve_range=True)

            # Save the resized rgb image
            output_rgb_filename = f"{sample}_{area}.tif"
            output_rgb_path = os.path.join(output_rgb_dir, output_rgb_filename)
            imwrite(output_rgb_path, rgb_img_resized.astype(np.dtype(rgb_img.dtype)))

            print(f"Processed and saved RGB: {output_rgb_path}")