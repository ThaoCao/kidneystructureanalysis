#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:59:48 2024

@author: thaocao
"""

import os
import numpy as np
from tifffile import imread
from skimage.transform import resize
import csv

rootdir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets'
datasets = ['Antibody_Mediated_Rejection', 'Lupus_Nephritis', 'Renal_Allograft', 'Normal_Kidney']
vessel_seg = 'vessel_segmentations/masks/cleaned'
tubule_seg = 'tubule_segmentations/masks/tubule_cleaned/cleaned'
glom_seg = 'glomeruli_segmentations/cleaned'
tissue_seg = 'tissue_composite_masks'
marker_seg = 'marker_segmentations/downsampled/'
collagen_seg = 'collagen_segmentations/downsampled/'

def clip_to_percentile(image):
    return np.where(image == 3, image, 0)

def is_sample_area_processed(csv_file, dataset, sample, area):
    if not os.path.exists(csv_file):
        return False
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        return any(row[0] == dataset and row[1] == sample and row[2] == area for row in reader)

def find_matching_image(image_list, sample, area):
    matches = [x for x in image_list if (sample in x) and (area in x)]
    return matches[0] if matches else None

def process_sample(dataset, sample, area):
    rdir_tissue = os.path.join(rootdir, dataset, 'tissue_composite_masks')
    rdir_vessel = os.path.join(rootdir, dataset, 'vessel_segmentations/masks/cleaned')
    rdir_tubule = os.path.join(rootdir, dataset, 'tubule_segmentations/masks/tubule_cleaned/cleaned')
    rdir_glom = os.path.join(rootdir, dataset, 'glomeruli_segmentations/cleaned')
    cd45_rdir = os.path.join(rootdir, dataset, 'collagen_segmentations/downsampled')
    ki67_rdir = os.path.join(rootdir, dataset, 'marker_segmentations/downsampled')
    mxa_rdir = os.path.join(rootdir, dataset, 'collagen_segmentations/downsampled')
    gzmb_rdir = os.path.join(rootdir, dataset, 'marker_segmentations/downsampled')
    claudin_rdir = os.path.join(rootdir, dataset, 'marker_segmentations/downsampled')

    # Find matching images
    tim = find_matching_image(os.listdir(rdir_tissue), sample, area)
    vessel_im = find_matching_image(os.listdir(rdir_vessel), sample, area)
    tubule_im = find_matching_image(os.listdir(rdir_tubule), sample, area)
    glom_im = find_matching_image(os.listdir(rdir_glom), sample, area)
    cd45_im = find_matching_image([f for f in os.listdir(cd45_rdir) if 'CD45' in f], sample, area)
    ki67_im = find_matching_image([f for f in os.listdir(ki67_rdir) if 'Ki67' in f], sample, area)
    mxa_im = find_matching_image([f for f in os.listdir(mxa_rdir) if 'MXA' in f], sample, area)
    gzmb_im = find_matching_image([f for f in os.listdir(gzmb_rdir) if 'GZMB' in f], sample, area)
    claudin_im = find_matching_image([f for f in os.listdir(claudin_rdir) if 'Claudin1' in f], sample, area)
    

    # Check if any image is missing
    missing_images = [img_name for img_name, img in [("tissue", tim), ("vessel", vessel_im), ("tubule", tubule_im), 
                                                     ("glomeruli", glom_im), ("CD45", cd45_im), ("Ki67", ki67_im),
                                                     ("MXA", mxa_im), ("GZMB", gzmb_im), ("Claudin", claudin_im)] if img is None]
    if missing_images:
        print(f"Missing image(s) for {dataset} - {sample}_{area}: {', '.join(missing_images)}")
        return None

    # Read images
    images = {}
    image_paths = {
        'vessel': os.path.join(rdir_vessel, find_matching_image(os.listdir(rdir_vessel), sample, area)),
        'tissue': os.path.join(rdir_tissue, find_matching_image(os.listdir(rdir_tissue), sample, area)),
        'tubule': os.path.join(rdir_tubule, find_matching_image(os.listdir(rdir_tubule), sample, area)),
        'glom': os.path.join(rdir_glom, find_matching_image(os.listdir(rdir_glom), sample, area)),
        'cd45': os.path.join(cd45_rdir, find_matching_image([f for f in os.listdir(cd45_rdir) if 'CD45' in f], sample, area)),
        'ki67': os.path.join(ki67_rdir, find_matching_image([f for f in os.listdir(ki67_rdir) if 'Ki67' in f], sample, area)),
        'mxa': os.path.join(mxa_rdir, find_matching_image([f for f in os.listdir(mxa_rdir) if 'MXA' in f], sample, area)),
        'gzmb': os.path.join(gzmb_rdir, find_matching_image([f for f in os.listdir(gzmb_rdir) if 'GZMB' in f], sample, area)),
        'claudin': os.path.join(claudin_rdir, find_matching_image([f for f in os.listdir(claudin_rdir) if 'Claudin1' in f], sample, area))
    }
    for key, path in image_paths.items():
        try:
            img = imread(path)
            if np.isnan(img).any():
                print(f"Warning: NaN values found in {key} image for {dataset} - {sample}_{area}")
                img = np.nan_to_num(img, nan=0.0)  # Replace NaN with 0
            images[key] = img
        except Exception as e:
            print(f"Error reading {key} image for {dataset} - {sample}_{area}: {str(e)}")
            return None
    
    # Resize images
    r, c = images['vessel'].shape
    for key in images:
        if key != 'vessel':
            try:
                resized = resize(images[key], (r, c), preserve_range=True, order=0 if key in ['tissue', 'tubule', 'glom'] else 1)
                if np.isnan(resized).any():
                    print(f"Warning: NaN values found after resizing {key} image for {dataset} - {sample}_{area}")
                    resized = np.nan_to_num(resized, nan=0.0)  # Replace NaN with 0
                images[key] = resized
            except Exception as e:
                print(f"Error resizing {key} image for {dataset} - {sample}_{area}: {str(e)}")
                return None

    
    # Resize images
    r, c = images['vessel'].shape
    for key in images:
        if key != 'vessel':
            images[key] = resize(images[key], (r, c), preserve_range=True, order=0 if key in ['tissue', 'tubule', 'glom'] else 1)

    # Clip marker images
    for key in ['cd45', 'ki67', 'mxa', 'gzmb', 'claudin']:
        images[f'{key}_clipped'] = clip_to_percentile(images[key])

    # Calculate areas and percentages
    tissue_area = np.sum(images['tissue'] > 0)
    tubule_area = np.sum(images['tubule'] > 0)
    glom_area = np.sum(images['glom'] > 0)
    vessel_area = np.sum(images['vessel'] > 0)
    interstitium_area = tissue_area - tubule_area - vessel_area - glom_area

    percentage_tubule = (tubule_area / tissue_area) * 100
    percentage_vessel = (vessel_area / tissue_area) * 100
    percentage_interstitium = (interstitium_area / tissue_area) * 100

    # Calculate marker percentages
    marker_percentages = {}
    for marker in ['cd45', 'ki67', 'mxa', 'gzmb', 'claudin']:
        for structure in ['tubule', 'vessel', 'interstitium']:
            if structure == 'interstitium':
                structure_mask = (images['tissue'] > 0) & (images['tubule'] == 0) & (images['vessel'] == 0) & (images['glom'] == 0)
            else:
                structure_mask = images[structure] > 0
            
            marker_area = np.sum((images[f'{marker}_clipped'] > 0) & structure_mask)
            structure_area = np.sum(structure_mask)
            
            marker_percentages[f'{marker}_{structure}'] = (marker_area / structure_area) * 100 if structure_area > 0 else 0

    marker_percentages['cd45_glom'] = np.sum((images['cd45_clipped'] > 0) & (images['glom'] > 0)) / glom_area * 100 if glom_area > 0 else 0

    # Calculate additional metrics
    num_tubules = len(np.unique(images['tubule'][images['tubule'] > 0]))
    num_glomeruli = len(np.unique(images['glom'][images['glom'] > 0]))
    total_area = np.sum(images['tissue'] > 0)
    tubule_density = num_tubules/total_area

    # Prepare row data
    row_data = [
        dataset, sample, area, percentage_tubule, percentage_vessel, percentage_interstitium,
        percentage_tubule / percentage_interstitium if percentage_interstitium != 0 else 0,
        percentage_vessel / percentage_interstitium if percentage_interstitium != 0 else 0,
        marker_percentages['cd45_tubule'], marker_percentages['cd45_vessel'], marker_percentages['cd45_interstitium'],
        marker_percentages['claudin_tubule'],
        marker_percentages['ki67_tubule'], marker_percentages['ki67_vessel'], marker_percentages['ki67_interstitium'],
        marker_percentages['gzmb_tubule'], marker_percentages['gzmb_vessel'], marker_percentages['gzmb_interstitium'],
        marker_percentages['mxa_tubule'], marker_percentages['mxa_vessel'], marker_percentages['mxa_interstitium'],
        marker_percentages['cd45_glom'],
        num_tubules,tubule_density, num_glomeruli, total_area
    ]

    return row_data

def main():
    csv_file = 'global_structure_analysis.csv'
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dataset', 'Sample', 'Area', 'Percentage tubule', 'Percentage_vessel', 'Percentage interstitium',
                             'Tubule to Interstitium ratio', 'Vessel to Interstitium ratio',
                             'Percentage CD45 in Tubule', 'Percentage CD45 in Vessel', 'Percentage CD45 in Interstitium',
                             'Percentage Claudin1 in Tubule',
                             'Percentage Ki67 in Tubule', 'Percentage Ki67 in Vessel', 'Percentage Ki67 in Interstitium',
                             'Percentage GZMB in Tubule', 'Percentage GZMB in Vessel', 'Percentage GZMB in Interstitium',
                             'Percentage MXA in Tubule', 'Percentage MXA in Vessel', 'Percentage MXA in Interstitium',
                             'Percentage CD45 in Glomeruli',
                             'Number of tubules', 'Tubule density', 'Number of glomeruli', 'Total area'])

    for dataset in datasets:
        rdir_vessel = os.path.join(rootdir, dataset, vessel_seg)
        vessel_ims = sorted(os.listdir(rdir_vessel))

        for sim in vessel_ims:
            sample = sim.split('_')[0]
            area = sim.split('_')[1].split('.')[0]

            if is_sample_area_processed(csv_file, dataset, sample, area):
                print(f"Skipping already processed sample: {dataset} - {sample}_{area}")
                continue

            row_data = process_sample(dataset, sample, area)
            if row_data:
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_data)
                print(f"Processed and added data for {dataset} - {sample}_{area}")
            else:
                print(f"Failed to process {dataset} - {sample}_{area}")

    print(f"All data written to {csv_file}")

if __name__ == "__main__":
    main()