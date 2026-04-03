#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:38:34 2024

@author: thaocao
Goal: merge all structures for all datasets 
"""
import os
import numpy as np
from skimage.io import imread, imsave
from tifffile import imread, imwrite
from skimage.transform import resize
from skimage.measure import label, regionprops

# Input directories for vessels, tubules, glomeruli, and cells
#TODO: change the directories
rootdir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets'
datasets = ['Antibody_Mediated_Rejection', 'Lupus_Nephritis', 'Normal_Kidney', 'Renal_Allograft']
vessel_seg = 'vessel_segmentations/masks/'
tubule_seg = 'tubule_segmentations/masks/'
glom_seg = 'glomeruli_segmentations'
tissue_seg = 'tissue_composite_masks'
cell_seg = 'wholeCell_segmentations_WS'
merge_directory = 'merged_segmentations_fullsize'
clean_output = 'cleaned'
fullsize = 'Corrected_DAPI_composites'

def count_tif_files(directory):
    """
    Count the number of .tif files in the given directory.
    """
    tif_files = [f for f in os.listdir(directory) if f.endswith('.tif')]
    return len(tif_files)

def print_clean_output_stats():
    """
    Count and print the number of .tif files in each clean_output directory
    for each dataset and segmentation type.
    """
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        for seg in [vessel_seg, tubule_seg, glom_seg]:
            clean_dir = os.path.join(rootdir, dataset, seg, clean_output)
            if os.path.exists(clean_dir):
                file_count = count_tif_files(clean_dir)
                print(f"  {seg}: {file_count} .tif files")
            else:
                print(f"  {seg}: Directory not found")

#QC the number of files in each output directory
print_clean_output_stats()

def find_matching_image(image_list, sample, area):
    matches = [x for x in image_list if (sample in x) and (area in x)]
    return matches[0] if matches else None

def clean_overlapping_labels(label_img, mask_img, threshold=0.95):
    cleaned_img = label_img.copy()
    for region in regionprops(label_img):
        label_mask = label_img == region.label
        overlap_ratio = np.sum(label_mask & mask_img) / np.sum(label_mask)
        if overlap_ratio > threshold:
            cleaned_img[label_mask] = 0
            print(f"Removed label {region.label} due to {overlap_ratio:.2f} overlap")
    return cleaned_img

for dataset in datasets:
    rdir_tissue = os.path.join(rootdir, dataset, tissue_seg)
    rdir_vessel = os.path.join(rootdir, dataset, vessel_seg)
    rdir_tubule = os.path.join(rootdir, dataset, tubule_seg)
    rdir_glom = os.path.join(rootdir, dataset, glom_seg)
    sdir_merge = os.path.join(rootdir, dataset, merge_directory)
    fullsize_dir = os.path.join(rootdir, dataset, fullsize)
    
    print(f"Processing {dataset}")
    
    # Create output directories
    for seg in [vessel_seg, tubule_seg, glom_seg]:
        clean_dir = os.path.join(rootdir, dataset, seg, clean_output)
        os.makedirs(clean_dir, exist_ok=True)
    
    # Create the merge directory if it doesn't exist
    os.makedirs(sdir_merge, exist_ok=True)
        
    tubule_ims = os.listdir(rdir_tubule)
    glom_ims = os.listdir(rdir_glom)
    vessel_ims = sorted(os.listdir(rdir_vessel))
    tims = os.listdir(rdir_tissue)

    for sim in vessel_ims:
        if not sim.endswith('.tif'):
            print(f"Skipping non-tif file: {sim}")
            continue
        sample = sim.split('_')[0]
        area = sim.split('_')[1].split('.')[0]
        sample_area = f"{sample}_{area}"
        rdir_cell = os.path.join(rootdir, dataset, cell_seg, sample, area)
        
        tim = find_matching_image(tims, sample, area)
        tubule_im = find_matching_image(tubule_ims, sample, area)
        glom_im = find_matching_image(glom_ims, sample, area)
        
        # Check if all required images are found
        if not all([tim, tubule_im, glom_im]):
            print(f"Skipping {sample_area} due to missing matching images")
            continue

        # Check if cell directory exists and contains files
        if not os.path.exists(rdir_cell) or not os.listdir(rdir_cell):
            print(f"Skipping {sample_area} due to missing or empty cell directory")
            continue
        
        cell_im = os.listdir(rdir_cell)[0]  
        try:
            vessel_img = imread(os.path.join(rdir_vessel, sim))
            timg = imread(os.path.join(rdir_tissue, tim))
            tubule_img = imread(os.path.join(rdir_tubule, tubule_im))
            glom_img = imread(os.path.join(rdir_glom, glom_im))
            cell_npy = np.load(os.path.join(rdir_cell, cell_im), allow_pickle=True)  
            cells = cell_npy.tolist()
            cell_img = cells['masks']
        except Exception as e:
            print(f"Error loading images for {sample_area}: {str(e)}")
            continue
        
        # Resize all images to match vessel_img size
        r, c = vessel_img.shape
        timg = resize(timg, (r, c), preserve_range=True, order=0).astype(np.uint8)
        tubule_img = resize(tubule_img, (r, c), preserve_range=True, order=0).astype(np.uint16)
        glom_img = resize(glom_img, (r, c), preserve_range=True, order=0).astype(np.uint16)
        cell_img = resize(cell_img, (r, c), preserve_range=True, order=0).astype(np.uint16)
        
        # Clean up the tubules and vessels images using the tissue mask 
        tubule_img = tubule_img * (timg // 255)
        vessel_img = vessel_img * (timg // 255)
 
        tsegs = np.where(tubule_img > 0, 1, 0)
        csegs = np.where(cell_img > 0, 1, 0)
        
        # Clean overlapping labels
        print(f"Processing {sample_area}")
        glom_img = clean_overlapping_labels(glom_img, tsegs)
        vessel_img = clean_overlapping_labels(vessel_img, tsegs)
        vessel_img = clean_overlapping_labels(vessel_img, csegs)
        
        # Resolve overlaps
        tubule_img = np.where((tubule_img + glom_img) > tubule_img, 0, tubule_img)
        glom_img = np.where((tubule_img + glom_img) > glom_img, glom_img, glom_img)
        glom_img = np.where((vessel_img + glom_img) > glom_img, glom_img, glom_img)
        vessel_img = np.where((vessel_img + glom_img) > vessel_img, 0, vessel_img)
        tubule_img = np.where((tubule_img + vessel_img) > tubule_img, tubule_img, tubule_img)
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
        
        # Save cleaned images (downsized)
        imwrite(os.path.join(rootdir, dataset, tubule_seg, clean_output, f"{sample_area}.tif"), tubule_img)
        imwrite(os.path.join(rootdir, dataset, glom_seg, clean_output, f"{sample_area}.tif"), glom_img)
        imwrite(os.path.join(rootdir, dataset, vessel_seg, clean_output, f"{sample_area}.tif"), vessel_img)
        print(f"Saved new clean masks for {sample_area}")
