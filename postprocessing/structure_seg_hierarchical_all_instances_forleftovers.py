#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:49:23 2024

@author: thaocao
Goal: specify a dataset and run cleaned-up segmentations for unprocessed samples
"""
import numpy as np
import os
from skimage.io import imread, imsave
from tifffile import imread, imwrite
from skimage.transform import resize
from skimage.measure import label, regionprops
from glob import glob

# Constants
ROOT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/'
DATASET = 'Renal_Allograft'
STRUCT = 'structure_analysis'
VESSEL_SEG = 'vessels'
TUBULE_SEG = 'tubules'
GLOM_SEG = 'glomeruli'
TISSUE_SEG = 'tissue'
MERGE_DIRECTORY = 'merged'
CLEAN_OUTPUT = 'cleaned'

def find_file(directory, search_string):
    for filename in os.listdir(directory):
        if search_string in filename and filename.endswith('.tif'):
            return filename
    return None

def clean_overlapping_labels(label_img, mask_img, threshold=0.95):
    cleaned_img = label_img.copy()
    for region in regionprops(label_img):
        label_mask = label_img == region.label
        overlap_ratio = np.sum(label_mask & mask_img) / np.sum(label_mask)
        if overlap_ratio > threshold:
            cleaned_img[label_mask] = 0
            print(f"Removed label {region.label} due to {overlap_ratio:.2f} overlap")
    return cleaned_img

def process_sample(sample_area_to_process):
    rdir_tissue = os.path.join(ROOT_DIR, DATASET, STRUCT, TISSUE_SEG)
    rdir_vessel = os.path.join(ROOT_DIR, DATASET, STRUCT, VESSEL_SEG)
    rdir_tubule = os.path.join(ROOT_DIR, DATASET, STRUCT, TUBULE_SEG)
    rdir_glom = os.path.join(ROOT_DIR, DATASET, STRUCT, GLOM_SEG)
    sdir_merge = os.path.join(ROOT_DIR, DATASET, STRUCT, MERGE_DIRECTORY)

    sample = sample_area_to_process.split('_')[0]
    area = sample_area_to_process.split('_')[1].split('.')[0]
    sample_area = f"{sample}_{area}"

    # Load images
    try:
        vessel_img = imread(os.path.join(rdir_vessel, sample_area_to_process))
        timg = imread(os.path.join(rdir_tissue, f"{sample_area}.tif"))
        tubule_img = imread(os.path.join(rdir_tubule, f"{sample_area}.tif"))
        glom_img = imread(os.path.join(rdir_glom, f"{sample_area}.tif"))
    except Exception as e:
        print(f"Error loading images for {sample_area}: {str(e)}")
        return

    # Resize all images to match vessel_img size
    r, c = vessel_img.shape
    timg = resize(timg, (r, c), preserve_range=True, order=0).astype(np.uint8)
    tubule_img = resize(tubule_img, (r, c), preserve_range=True, order=0).astype(np.uint16)
    glom_img = resize(glom_img, (r, c), preserve_range=True, order=0).astype(np.uint16)

    # Clean up the tubules and vessels images using the tissue mask; this step has been QCed extensively! --> should be an accurate measurement of the whole tissue space!
    tubule_img = tubule_img * (timg // 255)
    vessel_img = vessel_img * (timg // 255)
    glom_img = glom_img * (timg // 255)
    tsegs = np.where(tubule_img > 0, 1, 0)

    # Clean overlapping labels
    print(f"Processing {sample_area}")
    glom_img = clean_overlapping_labels(glom_img, tsegs)
    vessel_img = clean_overlapping_labels(vessel_img, tsegs)

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
    merged_image = np.zeros((r, c), dtype=np.uint8)
    merged_image[interstitium_img > 0] = 10
    merged_image[tubule_img > 0] = 100
    merged_image[vessel_img > 0] = 200
    merged_image[glom_img > 0] = 255

    # Save merged image output 
    merged_file_path = os.path.join(sdir_merge, f"{sample_area}.tif")
    imwrite(merged_file_path, merged_image)

    # Save cleaned images
    imwrite(os.path.join(ROOT_DIR, DATASET, STRUCT, TUBULE_SEG, CLEAN_OUTPUT, f"{sample_area}.tif"), tubule_img)
    imwrite(os.path.join(ROOT_DIR, DATASET, STRUCT, GLOM_SEG, CLEAN_OUTPUT, f"{sample_area}.tif"), glom_img)
    imwrite(os.path.join(ROOT_DIR, DATASET, STRUCT, VESSEL_SEG, CLEAN_OUTPUT, f"{sample_area}.tif"), vessel_img)
    print(f"Saved new clean masks for {sample_area}")

def main():
    # Create output directories
    for seg in [VESSEL_SEG, TUBULE_SEG, GLOM_SEG]:
        clean_dir = os.path.join(ROOT_DIR, DATASET, STRUCT, seg, CLEAN_OUTPUT)
        os.makedirs(clean_dir, exist_ok=True)

    # Create the merge directory if it doesn't exist
    sdir_merge = os.path.join(ROOT_DIR, DATASET,STRUCT, MERGE_DIRECTORY)
    os.makedirs(sdir_merge, exist_ok=True)

    # Get all sample areas that haven't been processed yet
    rdir_vessel = os.path.join(ROOT_DIR, DATASET,STRUCT, VESSEL_SEG)
    processed_samples = set(os.path.basename(f).split('.')[0] for f in glob(os.path.join(sdir_merge, '*.tif')))
    print(processed_samples)
    all_samples = set(os.path.basename(f).split('.')[0] for f in glob(os.path.join(rdir_vessel, '*.tif')))
    print(all_samples)
    samples_to_process = all_samples - processed_samples
    print(samples_to_process)
    for sample_area in samples_to_process:
        sample_area_to_process = f"{sample_area}.tif"
        process_sample(sample_area_to_process)

if __name__ == "__main__":
    main()