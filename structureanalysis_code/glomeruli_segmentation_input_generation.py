#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:15:46 2025

@author: thaocao
"""
import os
from tifffile import imread, imwrite
import numpy as np
from PIL import Image
#from skimage.transform import resize
import argparse 
import pickle as pkl

rootdir = '/project/mclark/SCAMPI_datasets/'
datasets = ['Normal_Kidney_60ch/']
save_output = 'glomeruli'
input_folder = 'ds10'  # Folder containing all marker images
markers = ['CD10_','DAPI', 'Claudin1', 'CD34']
for dataset in datasets:
    marker_paths = {marker: os.path.join(rootdir, dataset, input_folder, marker) for marker in markers}
    
    # Get sorted list of images for each marker
    marker_images = {marker: sorted(os.listdir(marker_paths[marker])) for marker in markers}
    
    # Create output directory for RGB images
    rgb_output_dir = os.path.join(rootdir, dataset, save_output)
    os.makedirs(rgb_output_dir, exist_ok=True)
    
    # Function to find matching image or return None
    def find_matching_image(image_list, sample, area):
        matches = [x for x in image_list if (sample in x) and (area in x)]
        return matches[0] if matches else None
    
# Function to resize image if it doesn't match CD10 size
# def resize_to_match(image, target_shape):
#     if image.shape != target_shape:
#         return resize(image, target_shape, preserve_range=True).astype(image.dtype)
#     return image
    
    # Function to convert 16-bit image to 8-bit
    def convert_16bit_to_8bit(image):
        image_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        return image_8bit

    
    for img in marker_images['CD10_']:
        sample, area = img.split('_')[:2]
        sample_area = f"{sample}_{area}"
        rgb_filename = f"{sample_area}_rgb.tif"
        output_path = os.path.join(rgb_output_dir, rgb_filename)
        
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"  Output already exists, skipping: {output_path}")
            continue
        print(f"Processing {sample_area}...")
        # Find matching images for all markers
        matching_images = {marker: find_matching_image(marker_images[marker], sample, area) for marker in markers}
        
        # Check if all matching images are found
        if all(matching_images.values()):
            # Read all images
            marker_data = {marker: imread(os.path.join(marker_paths[marker], matching_images[marker])) for marker in markers}
            
            # Get the shape of CD10 image
            cd10_shape = marker_data['CD10_'].shape
            
            # Resize other images if they don't match CD10 size and convert to 8-bit
            for marker in markers:
            #     if marker != 'CD10_':
            #         marker_data[marker] = resize_to_match(marker_data[marker], cd10_shape)
                marker_data[marker] = convert_16bit_to_8bit(marker_data[marker])
            # Generate RGB composite
            R_GL = np.maximum(marker_data['Claudin1'], 0)
            G_GL = np.maximum(marker_data['CD10_'], 0)
            B_GL = np.maximum(marker_data['CD34'], 0)
            
            RGB_GL = np.zeros((*cd10_shape, 3), dtype='uint8')
            RGB_GL[:,:,0] = R_GL
            RGB_GL[:,:,1] = G_GL
            RGB_GL[:,:,2] = B_GL
            
            # Save the RGB file
            rgb_filename = f"{sample_area}_rgb.tif"
            imwrite(os.path.join(rgb_output_dir, rgb_filename), RGB_GL)
            print(f"Saved RGB image: {rgb_filename}")
        else:
            print(f"Missing matching images for {sample_area}")
    
    
