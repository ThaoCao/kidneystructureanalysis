#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:39:18 2024

@author: thaocao
generate RGB composite for glomeruli segmentation with CD10 (green), CD34 (blue), and Claudin1 (red)
"""
import os
from tifffile import imread, imwrite
import numpy as np
from skimage.transform import resize
import argparse 
import pickle as pkl

rootdir = '/project/mclark/SCAMPI_datasets'
datasets = ['NK2', 'IgG4', 'AAV']
norm_comp = 'Normalized_composites'
ds10 = 'ds10'
markers = ['CD10_']

def downsample_and_save(img_path, output_path):
    img = imread(img_path)
    r, c = np.shape(img)
    ds_img = resize(img, (r//10, c//10), order=1, preserve_range=True)
    imwrite(output_path, ds_img.astype(np.uint16))

# Generate ds10 folders and downsample images
for dataset in datasets:
    for marker in markers:
        ds10_marker_path = os.path.join(rootdir, dataset, ds10, marker)
        os.makedirs(ds10_marker_path, exist_ok=True)
        
        norm_comp_path = os.path.join(rootdir, dataset, norm_comp)
        for sample in os.listdir(norm_comp_path):
            sample_path = os.path.join(norm_comp_path, sample)
            if os.path.isdir(sample_path):
                for area in os.listdir(sample_path):
                    area_path = os.path.join(sample_path, area)
                    if os.path.isdir(area_path):
                        for file in os.listdir(area_path):
                            if file.startswith(marker):
                                img_path = os.path.join(area_path, file)
                                output_filename = f"{sample}_{area}.tif"
                                output_path = os.path.join(ds10_marker_path, output_filename)
                                if not os.path.exists(output_path):
                                    downsample_and_save(img_path, output_path)
                                    print(f"Downsampled and saved: {output_path}")
    # Create output directory for RGB images
    rgb_output_dir = os.path.join(rootdir,dataset,ds10,'rgb_glom')
    os.makedirs(rgb_output_dir, exist_ok=True)

# Rest of the script remains the same
marker_paths = {marker: os.path.join(rootdir, dataset, ds10, marker) for marker in markers}

# Get sorted list of images for each marker
marker_images = {marker: sorted(os.listdir(marker_paths[marker])) for marker in markers}


# # Function to find matching image or return None
# def find_matching_image(image_list, sample, area):
#     matches = [x for x in image_list if (sample in x) and (area in x)]
#     return matches[0] if matches else None

# # Function to resize image if it doesn't match CD10 size
# def resize_to_match(image, target_shape):
#     if image.shape != target_shape:
#         return resize(image, target_shape, preserve_range=True).astype(image.dtype)
#     return image

# for img in marker_images['CD138']:
#     sample, area = img.split('_')[:2]
#     sample_area = f"{sample}_{area}"
    
#     # Find matching images for all markers
#     matching_images = {marker: find_matching_image(marker_images[marker], sample, area) for marker in markers}
    
#     # Check if all matching images are found
#     if all(matching_images.values()):
#         # Read all images
#         marker_data = {marker: imread(os.path.join(marker_paths[marker], matching_images[marker])) for marker in markers}
        
#         # Get the shape of CD10 image
#         cd10_shape = marker_data['CD138'].shape
        
#         # Resize other images if they don't match CD10 size
#         for marker in markers:
#             if marker != 'CD138':
#                 marker_data[marker] = resize_to_match(marker_data[marker], cd10_shape)
        
#         # Generate RGB composite
#         R_GL = np.maximum(marker_data['Claudin1'], marker_data['DAPI'])
#         G_GL = np.maximum(marker_data['CD10'], marker_data['DAPI'])
#         B_GL = np.maximum(marker_data['CD34'], marker_data['DAPI'])
        
#         RGB_GL = np.zeros((*cd10_shape, 3), dtype='uint8')
#         RGB_GL[:,:,0] = R_GL
#         RGB_GL[:,:,1] = G_GL
#         RGB_GL[:,:,2] = B_GL
        
#         # Save the RGB file
#         rgb_filename = f"{sample_area}_rgb.tif"
#         imwrite(os.path.join(rgb_output_dir, rgb_filename), RGB_GL)
#         print(f"Saved RGB image: {rgb_filename}")
#     else:
#         print(f"Missing matching images for {sample_area}")
