#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:30:20 2025

@author: thaocao
This script processes structure marker overlays and generates summed images for tubule segmentation input generation on 8-bit single channels inputs.
"""
import os
import numpy as np
from tifffile import imread, imwrite

rootdir = '/project/mclark/SCAMPI_datasets/'
datasets = ['Lupus_Nephritis_60ch']
save_output = 'tubules'
input_folder = 'ds10'  # Folder containing all 4 marker images
marker_tubules_list = ['MUC1', 'Claudin1', 'CD138', 'CD10_']

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset}")
    print(f"{'='*80}")
    
    savedir = os.path.join(rootdir, dataset, save_output)
    if not os.path.exists(savedir): 
        os.makedirs(savedir)

    # Get the first marker directory to get the list of all sample files
    first_marker_dir = os.path.join(rootdir, dataset, input_folder, marker_tubules_list[0])
    
    if not os.path.exists(first_marker_dir):
        print(f"Error: Marker directory not found: {first_marker_dir}")
        continue
    
    # Get all .tif files from the first marker directory
    all_files = [f for f in os.listdir(first_marker_dir) if f.endswith('.tif')]
    
    if not all_files:
        print(f"No .tif files found in {first_marker_dir}")
        continue
    
    print(f"Found {len(all_files)} samples to process")
    
    # Process each sample
    for sample_file in all_files:
        print(f"\nProcessing sample: {sample_file}")
        output_path = os.path.join(savedir, sample_file)    
        # Skip if output already exists
        if os.path.exists(output_path):
            print(f"  Output already exists, skipping: {output_path}")
            continue
        # Load all marker images for this sample
        marker_array_list = []
        reference_shape = None
        all_markers_found = True
        
        for marker in marker_tubules_list:
            marker_path = os.path.join(rootdir, dataset, input_folder, marker)
            img_path = os.path.join(marker_path, sample_file)
            
            if not os.path.exists(img_path):
                print(f"  Warning: File not found for {marker}: {img_path}")
                all_markers_found = False
                break
            
            img = imread(img_path)
            
            # Set reference shape from first image
            if reference_shape is None:
                reference_shape = img.shape
            
            # Check if all images have the same shape
            if img.shape != reference_shape:
                print(f"  Warning: {marker} has different shape {img.shape}, expected {reference_shape}")
            
            marker_array_list.append(img)
            print(f"  Loaded {marker}: shape {img.shape}")
        
        if not all_markers_found:
            print(f"  Skipping {sample_file} due to missing markers")
            continue
        
        # Stack all marker images
        marker_array = np.stack(marker_array_list, axis=0)
        
        # Sum all channels
        sum_of_channels = np.sum(marker_array, axis=0)
        
        # Normalize
        norm_sum = sum_of_channels / (255 * len(marker_tubules_list))
        
        # Avoid division by zero
        max_val = np.max(norm_sum)
        if max_val > 0:
            norm_sum_2 = 255 * norm_sum / max_val
        else:
            norm_sum_2 = norm_sum
        
        # Save output with the same filename
        
        imwrite(output_path, norm_sum_2.astype(np.uint8))
        
        print(f"  Saved to: {output_path}")
    
    print(f"\nCompleted processing dataset: {dataset}")

print("\n" + "="*80)
print("All datasets processed!")
print("="*80)
