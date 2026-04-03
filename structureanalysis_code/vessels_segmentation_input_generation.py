#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:39:18 2024

@author: thaocao
generate vessel input
"""
import os
from tifffile import imread, imwrite
import numpy as np
from PIL import Image
from skimage.transform import resize
import argparse 
import pickle as pkl

rootdir = '/project/mclark/SCAMPI_datasets/'
datasets = ['NK2']
markers = ['CD138', 'CD34']

for dataset in datasets:
    marker_paths = {marker: os.path.join(rootdir, dataset, 'ds10', marker) for marker in markers}
    
    # Check if marker directories exist
    missing_dirs = [marker for marker in markers if not os.path.exists(marker_paths[marker])]
    if missing_dirs:
        print(f"Missing directories for {dataset}: {missing_dirs}")
        continue
    
    # Get sorted list of images for each marker
    marker_images = {marker: sorted([f for f in os.listdir(marker_paths[marker]) if f.endswith('.tif')]) for marker in markers}
    
    # Create output directory
    output_dir = os.path.join(rootdir, dataset, 'vessels_segmentation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Function to find matching image or return None
    def find_matching_image(image_list, sample, area):
        matches = [x for x in image_list if (sample in x) and (area in x)]
        return matches[0] if matches else None
    
    # Function to resize image if it doesn't match CD34 size
    def resize_to_match(image, target_shape):
        if image.shape != target_shape:
            return resize(image, target_shape, preserve_range=True).astype(image.dtype)
        return image
    
    # Function to convert 16-bit image to 8-bit
    def convert_16bit_to_8bit(image):
        if image.max() > image.min():  # Avoid division by zero
            image_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_8bit = np.zeros_like(image, dtype=np.uint8)
        return image_8bit
    
    print(f"Processing dataset: {dataset}")
    print(f"CD34 images found: {len(marker_images['CD34'])}")
    print(f"CD138 images found: {len(marker_images['CD138'])}")
    
    for img in marker_images['CD34']:
        try:
            # Extract sample and area from filename
            parts = img.split('_')
            if len(parts) < 2:
                print(f"Skipping {img} - cannot extract sample and area")
                continue
                
            sample, area = parts[0], parts[1]
            sample_area = f"{sample}_{area}"
            
            # Find matching images for all markers
            matching_images = {marker: find_matching_image(marker_images[marker], sample, area) for marker in markers}
            
            # Check if all matching images are found
            if all(matching_images.values()):
                # Read all images
                marker_data = {}
                for marker in markers:
                    img_path = os.path.join(marker_paths[marker], matching_images[marker])
                    marker_data[marker] = imread(img_path)
                
                # Get the shape of CD34 image
                cd34_shape = marker_data['CD34'].shape
                
                # Resize CD138 to match CD34 if needed
                marker_data['CD138'] = resize_to_match(marker_data['CD138'], cd34_shape)
                
                # Generate CD34 subtracting CD138
                
                subtract_img = marker_data['CD34'] - marker_data['CD138']
                subtract_img[subtract_img < 0] = 0
                
                #subtract_img = subtract_img.astype(marker_data['CD34'].dtype)
                
                # Save the subtracted image with .tif extension
                output_filename = f"{sample_area}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Check if file already exists
                if os.path.exists(output_path):
                    print(f"File {output_filename} already exists. Skipping...")
                    continue
                
                imwrite(output_path, subtract_img)
                print(f"Saved subtracted image: {output_filename}")
                
            else:
                missing_markers = [marker for marker in markers if not matching_images[marker]]
                print(f"Missing matching images for {sample_area} - missing markers: {missing_markers}")
                
        except Exception as e:
            print(f"Error processing {img}: {str(e)}")
            continue

print("Processing complete!")
