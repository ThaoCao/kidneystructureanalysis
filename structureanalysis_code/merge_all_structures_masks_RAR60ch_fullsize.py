#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 20:04:28 2023

@author: thaocao
this script is to merge masks of all structures: vessels, tubules, glomeruli
"""

import os
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize
from skimage.measure import label, regionprops

# Input directories for vessels, tubules, glomeruli

rootdir = '/project/mclark/SCAMPI_datasets/'
datasets = ['Duke_TCMR_60ch']
vessel_seg = 'vessels'
tubule_seg = 'tubules'
glom_seg = 'Glomeruli_manual_GT_mask_upsized'
tissue_seg = 'tissue_composite_masks'
merge_directory = 'merged_segmentations_fullsize'
fullsize = 'Glomeruli_manual_GT_mask_upsized'
clean_output = 'cleaned'

def find_matching_image(image_list, sample, area):
    matches = [x for x in image_list if (sample in x) and (area in x)]
    return matches[0] if matches else None

def clean_overlapping_labels(label_img, mask_img, threshold=0.8):
    cleaned_img = label_img.copy()
    for region in regionprops(label_img):
        label_mask = label_img == region.label
        overlap_ratio = np.sum(label_mask & mask_img) / np.sum(label_mask)
        if overlap_ratio > threshold:
            cleaned_img[label_mask] = 0
            print(f"Removed label {region.label} due to {overlap_ratio:.2f} overlap")
    return cleaned_img

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset}")
    print(f"{'='*80}")
    
    rootdir = os.path.join('/project/mclark/SCAMPI_datasets/', dataset)
    rdir_tissue = os.path.join(rootdir, tissue_seg)
    rdir_vessel = os.path.join(rootdir, vessel_seg)
    rdir_tubule = os.path.join(rootdir, tubule_seg)
    rdir_glom = os.path.join(rootdir, glom_seg)
    sdir_merge = os.path.join(rootdir, merge_directory)
    fullsize_dir = os.path.join(rootdir, fullsize)

    # Create output directories
    for seg in [vessel_seg, tubule_seg]:
        clean_dir = os.path.join(rootdir, seg, clean_output)
        os.makedirs(clean_dir, exist_ok=True)
        print('created cleaned output')

    # Create the merge directory if it doesn't exist
    os.makedirs(sdir_merge, exist_ok=True)
        
    tubule_ims = os.listdir(rdir_tubule)
    glom_ims = os.listdir(rdir_glom)
    vessel_ims = sorted(os.listdir(rdir_vessel))
    tims = os.listdir(rdir_tissue)

    for sim in tubule_ims:
            if not sim.endswith('.tif'):
                print(f"Skipping non-tif file: {sim}")
                continue
            sample = sim.split('_')[0]
            area = sim.split('_')[1].split('.')[0]
            sample_area = f"{sample}_{area}"
            
            tim = find_matching_image(tims, sample, area)
            vessel_im = find_matching_image(vessel_ims, sample, area)
            glom_im = find_matching_image(glom_ims, sample, area)
            
            # Check if all required images are found
            if not all([tim, vessel_im, glom_im]):
                print(f"Skipping {sample_area} due to missing matching images")
                continue

            try:
                vessel_img = imread(os.path.join(rdir_vessel, vessel_im))
                timg = imread(os.path.join(rdir_tissue, tim))
                tubule_img = imread(os.path.join(rdir_tubule, sim))
                glom_img = imread(os.path.join(rdir_glom, glom_im))
                # Check if all images match in size, resize to full size glom img if needed
                target_r, target_c = glom_img.shape
                r, c = target_r, target_c
                if vessel_img.shape != (target_r, target_c):
                    vessel_img = resize(vessel_img, (target_r, target_c), preserve_range=True, order=0).astype(vessel_img.dtype)
                if timg.shape != (target_r, target_c):
                    timg = resize(timg, (target_r, target_c), preserve_range=True, order=0).astype(timg.dtype)
                if tubule_img.shape != (target_r, target_c):
                    tubule_img = resize(tubule_img, (target_r, target_c), preserve_range=True, order=0).astype(tubule_img.dtype)
                
            except Exception as e:
                print(f"Error loading images for {sample_area}: {str(e)}")
                continue
            
            
            # Clean up the tubules and vessels images using the tissue mask (8-bit, 255 for tissue)
            tubule_img = tubule_img * (timg > 0).astype(tubule_img.dtype)
            vessel_img = vessel_img * (timg > 0).astype(vessel_img.dtype)

            # Use a cell-size cutoff to remove spurious tubule labels that are actually cells
            tubule_props = regionprops(label(tubule_img))
            for prop in tubule_props:
                if prop.area < 10000:  # calculated for an average cell with 10um diameter = 6.64 pixels in downsized tubule images
                    tubule_img[tubule_img == prop.label] = 0

            # Create binary mask of tubules for overlap checking
            tsegs = np.where(tubule_img > 0, 1, 0)
            gsegs = np.where(glom_img > 0, 1, 0)
            # Clean overlapping labels
            print(f"Processing {sample_area}")
            tubule_img = clean_overlapping_labels(tubule_img, gsegs)
            vessel_img = clean_overlapping_labels(vessel_img, gsegs)
            vessel_img = clean_overlapping_labels(vessel_img, tsegs)
            
            # Resolve overlaps
            tubule_img = np.where((tubule_img + glom_img) > tubule_img, 0, tubule_img)
            vessel_img = np.where((vessel_img + glom_img) > vessel_img, 0, vessel_img)
            vessel_img = np.where((vessel_img + tubule_img) > vessel_img, 0, vessel_img)
            
            # Create interstitium image
            interstitium_img = np.zeros((r, c), dtype=np.uint8)
            interstitium_img[(timg > 0) & (tubule_img == 0) & (vessel_img == 0) & (glom_img == 0)] = 1
            
            # Create merged image
            print(f"Creating merged mask for {sample_area}")
            merged_image = np.zeros((r, c), dtype=np.uint8)
            merged_image[interstitium_img > 0] = 10
            merged_image[tubule_img > 0] = 100
            merged_image[vessel_img > 0] = 200
            merged_image[glom_img > 0] = 255
            
            # Resize merged_image to match the original full-size composite
            if os.path.exists(fullsize_dir):
                matched_fullsize_imgs = [f for f in os.listdir(fullsize_dir) 
                                        if sample in f and area in f and f.endswith('.tif')]
                
                if matched_fullsize_imgs:
                    # Take the first match
                    fullsize_img_path = os.path.join(fullsize_dir, matched_fullsize_imgs[0])
                    try:
                        fullsize_img = imread(fullsize_img_path)
                        
                        # Get dimensions of the full-size image
                        fullsize_r, fullsize_c = fullsize_img.shape[:2]
                        
                        # Resize the merged image to match full-size dimensions
                        merged_image_resized = resize(merged_image, (fullsize_r, fullsize_c), 
                                                    order=0, preserve_range=True).astype(np.uint8)
                        
                        print(f"Resized merged image from {r}x{c} to {fullsize_r}x{fullsize_c}")
                        
                        # Use the resized image for saving
                        merged_image = merged_image_resized
                        
                    except Exception as e:
                        print(f"Error loading or resizing full-size image for {sample_area}: {str(e)}")
                        print(f"Using original size for merged image")
                else:
                    print(f"No matching full-size image found for {sample_area} in {fullsize_dir}")
                    print(f"Using original size for merged image")
            else:
                print(f"Full-size directory not found: {fullsize_dir}")
                print(f"Using original size for merged image")

            # Save merged image
            merged_file_path = os.path.join(sdir_merge, f"{sample_area}.tif")
            imwrite(merged_file_path, merged_image)
            print(f"Saved merged mask to {merged_file_path}")
            
            # Save cleaned images
            tubule_clean_path = os.path.join(rootdir, tubule_seg, clean_output, f"{sample_area}.tif")
            #glom_clean_path = os.path.join(rootdir, glom_seg, clean_output, f"{sample_area}.tif")
            vessel_clean_path = os.path.join(rootdir, vessel_seg, clean_output, f"{sample_area}.tif")
            
            imwrite(tubule_clean_path, tubule_img)
            #imwrite(glom_clean_path, glom_img)
            imwrite(vessel_clean_path, vessel_img)
            
            print(f"Saved cleaned tubule mask to {tubule_clean_path}")
            #print(f"Saved cleaned glomeruli mask to {glom_clean_path}")
            print(f"Saved cleaned vessel mask to {vessel_clean_path}")
    
    # Now merge all cleaned structures masks for this dataset
    print(f"\n{'='*80}")
    print(f"Merging cleaned structures for dataset: {dataset}")
    print(f"{'='*80}")
    
    tubule_clean_dir = os.path.join(rootdir, tubule_seg, clean_output)
    vessel_clean_dir = os.path.join(rootdir, vessel_seg, clean_output)
    merged_cleaned_dir = os.path.join(rootdir, 'merged_all_structures_cleaned')
    os.makedirs(merged_cleaned_dir, exist_ok=True)
    
    tubule_files = [f for f in os.listdir(tubule_clean_dir) if f.endswith('.tif')]
    
    for tubule_file in tubule_files:
        sample_area = tubule_file.replace('.tif', '')
        
        try:
            tubule_img = imread(os.path.join(tubule_clean_dir, tubule_file))
            vessel_img = imread(os.path.join(vessel_clean_dir, tubule_file))
            glom_img = imread(os.path.join(rdir_glom, tubule_file))
            tissue_img = imread(os.path.join(rdir_tissue, tubule_file))
            
            r, c = glom_img.shape
            
            # Create interstitium from tissue mask
            interstitium_img = np.zeros((r, c), dtype=np.uint8)
            interstitium_img[(tissue_img > 0) & (tubule_img == 0) & (vessel_img == 0) & (glom_img == 0)] = 1
            
        except Exception as e:
            print(f"Error loading images for {sample_area}: {str(e)}")
            continue
            
        # Create merged image
        print(f"Creating merged cleaned mask for {sample_area}")
        merged_image = np.zeros((r, c), dtype=np.uint8)
        merged_image[interstitium_img > 0] = 10
        merged_image[tubule_img > 0] = 100
        merged_image[vessel_img > 0] = 200
        merged_image[glom_img > 0] = 255 

        # Save merged image
        merged_file_path = os.path.join(merged_cleaned_dir, f"{sample_area}.tif")
        imwrite(merged_file_path, merged_image)
        print(f"Saved merged cleaned mask to {merged_file_path}")
    
    print(f"\nCompleted processing dataset: {dataset}")

print("\n" + "="*80)
print("All datasets processed!")
print("="*80)
            
