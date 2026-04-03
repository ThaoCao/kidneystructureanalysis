#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:19:36 2024

@author: thaocao

"""

import os
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize

rootdir = '/project/mclark/SCAMPI_datasets'
datasets = ['Lupus_Nephritis_60ch']  #TODO: edit the dataset list
markers = ['CD34', 'CD10_', 'CD138', 'Claudin1', 'MUC1'] 
compdir = 'Normalized_composites/'
base_save_dir = 'ds10/'

for dataset in datasets:
    dataset_dir = os.path.join(rootdir, dataset, compdir)
    

    for marker in markers:
        save_dir = os.path.join(base_save_dir, f'{marker}/')
        output_dataset_dir = os.path.join(rootdir, dataset, save_dir)
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        print(f"Processing {dataset} - {marker}")
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist. Skipping...")
            continue
            
        # Iterate through samples
        for sample in os.listdir(dataset_dir):
            sample_dir = os.path.join(dataset_dir, sample)
            
            # Skip if not a directory
            if not os.path.isdir(sample_dir):
                continue
                
            # Iterate through areas
            for area in os.listdir(sample_dir):
                area_dir = os.path.join(sample_dir, area)
                
                # Skip if not a directory
                if not os.path.isdir(area_dir):
                    continue
                
                # Look for files that start with the current marker name
                marker_files = [f for f in os.listdir(area_dir) if f.startswith(marker) and f.endswith('.tif')]
                
                if not marker_files:
                    print(f"No {marker} file found in {area_dir}")
                    continue
                
                marker_file = marker_files[0]
                marker_path = os.path.join(area_dir, marker_file)
                output_filename = f"{sample}_{area}.tif"
                output_path = os.path.join(output_dataset_dir, output_filename)
                
                # Check if the output file already exists
                if os.path.exists(output_path):
                    print(f"Output file {output_filename} already exists. Skipping...")
                    continue
                
                try:
                    # Read and resize image
                    img = imread(marker_path)
                    r, c = np.shape(img)
                    img_resized = resize(img, (r//10, c//10), order=1, anti_aliasing=True, preserve_range=True)
                    
                    # Save resized image in the appropriate dataset folder
                    imwrite(output_path, img_resized.astype(np.uint8))
                    print(f"Saved resized {marker} image to {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {marker_path}: {str(e)}")
                    continue

print("Processing complete!")
