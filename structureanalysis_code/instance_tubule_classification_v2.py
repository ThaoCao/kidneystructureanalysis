#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 18:16:39 2025

@author: thaocao
Instance tubules classifications after creating the tubules csv based on: 'size', 'CD10', 'MUC1', 'CD45', 'Claudin1', 'CD10_freq', 'MUC1_freq', 'CD45_freq', 'Claudin1_freq'

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.ndimage import distance_transform_edt

# Define input and output directories
ROOT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/'
DATASETS = ['Antibody_Mediated_Rejection', 'Lupus_Nephritis', 'Renal_Allograft','Normal_Kidney']
CSV_DIR = 'structural_analysis_instances/infiltration/tubules/'
TUBULES_SEG = 'structural_analysis_instances/tubules/'
MARKER_DIR = 'ds10'
MARKERS = ['CD10', 'MUC1', 'Claudin1', 'CD45']
OUTPUT_DIR = 'structural_analysis_instances/tubule_classes/'
VESSEL_ITIS = 'structural_analysis_instances/infiltration/vessels/'
VESSELS_SEG = 'structural_analysis_instances/vessels/'
VESSEL_ITIS = 'structural_analysis_instances/infiltration/vessels/' #csv
OUTPUT = 'structural_analysis_instances/vesselitis'

# Constants for classification
INTENSITY_THRESHOLD = 20
FREQUENCY_THRESHOLD = 10
PERCENTILE_THRESHOLD = 95

def process_tubules(DATASET):
    """
    Processes tubule segmentation, calculates inter-tubule distances,
    classifies tubules based on marker intensities, and assigns colors.
    """
    # Define paths
    tubules_seg_dir = os.path.join(ROOT_DIR, DATASET, TUBULES_SEG)
    csv_dir = os.path.join(ROOT_DIR, DATASET, CSV_DIR)
    output_class_dir = os.path.join(ROOT_DIR, DATASET, OUTPUT_DIR)
    
    # Create output directory
    os.makedirs(output_class_dir, exist_ok=True)

    # Iterate over tubule segmentation files
    for filename in os.listdir(tubules_seg_dir):
        if not filename.endswith('.tif'):
            continue

        filename_path = os.path.join(tubules_seg_dir, filename)
        tubules_mask = imread(filename_path)
        sample_area = os.path.splitext(filename)[0]
        csv_path = os.path.join(csv_dir, f"{sample_area}_tubules.csv")

        # Check if CSV file exists; create if not
        if not os.path.exists(csv_path):
            print(f"CSV file not found for {sample_area} in dataset {DATASET}. Creating...")
            instance_ids = np.unique(tubules_mask)
            instance_ids = instance_ids[instance_ids != 0]  # Exclude background (0)
            df = pd.DataFrame({'ID': instance_ids})
            df.to_csv(csv_path, index=False)
        
        # Prepare CSV
        df = prepare_tubule_dataframe(csv_path, tubules_mask, DATASET, sample_area)
        
        # Inter-tubule distances, classification
        df = calculate_inter_tubule_distances(tubules_mask, df)
        df = classify_tubules(tubules_mask, df, DATASET, sample_area)
        
        # Update CSV
        df.to_csv(csv_path, index=False)

        # Color-coded image
        output_path = os.path.join(output_class_dir, f"{sample_area}.tif")
        assign_tubule_colors(tubules_mask, csv_path, output_path)
        
        print(f"Processed tubule segmentation for {sample_area} in dataset {DATASET}.")

def prepare_tubule_dataframe(csv_path, tubules_mask, DATASET, sample_area):
    """
    Loads the CSV, calculates tubule size, adds classification columns,
    handles file existence and dimension checks.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Error reading CSV file for {sample_area} in dataset {DATASET}. Skipping. Error: {e}")
        return pd.DataFrame()

    # Add new columns to the DataFrame if they don't exist
    for column in ['Proximal', 'Distal', 'Stressed','Inflamed', 'Unclassified', 'Dist', 'Size']:
        if column not in df.columns:
            df[column] = 0  # Initialize with 0

    # Calculate size for each instance
    instance_ids = np.unique(tubules_mask)
    instance_ids = instance_ids[instance_ids != 0]
    for instance_id in instance_ids:
        instance_mask = (tubules_mask == instance_id)
        size = np.sum(instance_mask)
        df.loc[df['ID'] == instance_id, 'Size'] = size

    return df

def calculate_inter_tubule_distances(tubules_mask, df):
    """
    Calculates the minimum distance from each tubule to another tubule.
    """
    if df.empty:  # Skip if the dataframe is empty
        return df
    
    for _, row in df.iterrows():
        instance_id = row['ID']
        
        # Create a binary mask for the current tubule instance
        instance_mask = (tubules_mask == instance_id)
        
        # Get the bounding box for the current instance
        rows = np.any(instance_mask, axis=1)
        cols = np.any(instance_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop the tubules_mask to the bounding box
        cropped_mask = tubules_mask[rmin:rmax+1, cmin:cmax+1]
        
        # Create a binary mask of all other tubules (excluding the current one)
        other_tubules_mask = (cropped_mask != instance_id) & (cropped_mask != 0)
        
        # Calculate the distance to the nearest tubule instance
        distance_map = distance_transform_edt(~other_tubules_mask)
        
        # Find the minimum distance to another tubule within the bounding box
        min_distance = np.min(distance_map[instance_mask[rmin:rmax+1, cmin:cmax+1]])
        
        # Update the 'Dist' column in the DataFrame
        df.loc[df['ID'] == instance_id, 'Dist'] = min_distance

    return df

def classify_tubules(tubules_mask, df, DATASET, sample_area):
    """
    Classifies tubules based on marker intensities.
    """
    if df.empty:  # Skip if the dataframe is empty
        return df
    
    tubules_seg_dir = os.path.join(ROOT_DIR, DATASET, TUBULES_SEG)
    csv_dir = os.path.join(ROOT_DIR, DATASET, CSV_DIR)
    
    # Iterate over each marker and perform classification
    for MARKER in MARKERS:
        marker_dir = os.path.join(ROOT_DIR, DATASET, MARKER_DIR, MARKER)
        marker_image_path = os.path.join(marker_dir, f"{sample_area}.tif")
        
        # Check if the marker image exists
        if not os.path.exists(marker_image_path):
            print(f"Warning: Marker image not found for {sample_area} in dataset {DATASET}, marker {MARKER}. Skipping.")
            continue
        
        try:
            marker_image = imread(marker_image_path)
            if marker_image.shape != tubules_mask.shape:
                print(f"Warning: Dimensions of marker image {marker_image.shape} do not match tubule mask {tubules_mask.shape} for {sample_area} in dataset {DATASET}, marker {MARKER}. Skipping.")
                continue
        except Exception as e:
            print(f"Warning: Error reading marker image for {sample_area} in dataset {DATASET}, marker {MARKER}. Skipping. Error: {e}")
            continue
    
        instance_ids = np.unique(tubules_mask)
        instance_ids = instance_ids[instance_ids != 0]
        
        for instance_id in instance_ids:
            instance_mask = (tubules_mask == instance_id)
            marker_pixels = marker_image[instance_mask]
            percentile_95 = np.percentile(marker_pixels, PERCENTILE_THRESHOLD)
            
            # Calculate frequency of the most common pixel value
            values, counts = np.unique(marker_pixels, return_counts=True)
            freq = counts.max()  # Frequency of the most common intensity
            
            # Classify based on the 95th percentile and frequency threshold
            if MARKER == 'CD10' and percentile_95 >= INTENSITY_THRESHOLD:
                df.loc[df['ID'] == instance_id, 'Proximal'] = 1
            elif MARKER == 'MUC1' and percentile_95 >= INTENSITY_THRESHOLD:
                df.loc[df['ID'] == instance_id, 'Distal'] = 1
            elif MARKER == 'Claudin1' and percentile_95 >= INTENSITY_THRESHOLD and freq > FREQUENCY_THRESHOLD:
                df.loc[df['ID'] == instance_id, 'Stressed'] = 1
            elif MARKER == 'CD45' and percentile_95 >= 40 and freq > FREQUENCY_THRESHOLD:
                df.loc[df['ID'] == instance_id, 'Inflamed'] = 1
        
            
    for _, row in df.iterrows():
        instance_id = row['ID']
        proximal = row['Proximal']
        distal = row['Distal']
        stressed = row['Stressed']
        inflamed = row['Inflamed']
        if proximal == 0 and distal == 0 and stressed == 0 and inflamed == 0:
            df.loc[df['ID'] == instance_id, 'Unclassified'] = 1
        elif proximal == 1 and distal == 1 and stressed == 1 and inflamed == 1: 
            df.loc[df['ID'] == instance_id, 'Unclassified'] = 1

    return df

def assign_tubule_colors(tubules_mask, csv_path, output_path):
    """
    Assigns colors to tubules based on their classification.
    """
    df = pd.read_csv(csv_path)
    rgb_image = np.zeros((tubules_mask.shape[0], tubules_mask.shape[1], 3), dtype=np.uint8)
    
    if df.empty: #TODO: test the effects of an empty df on running the remaining steps, and potentially log a warning to the user
        print(f"Warning: dataframe is empty")
        imwrite(output_path, rgb_image) #save a black image if no tubules
        return #skip to the next iteration

    # Assign colors based on classification
    for _, row in df.iterrows():
        instance_id = row['ID']
        proximal = row['Proximal']
        distal = row['Distal']
        stressed = row['Stressed']
        inflamed = row['Inflamed']
            
        #TODO: use a new color scheme to accomodate more classifications: proximal and inflamed, distal and inflamed, stressed and inflamed
        if proximal == 1 and distal == 1:
            color = [128, 128, 128]  # Gray for Proximal and Distal
        elif proximal == 1 and stressed == 1:
            color = [255, 255, 0]    # Yellow for Proximal and Stressed
        elif distal == 1 and stressed == 1:
            color = [255, 0, 255]    # Magenta for Distal and Stressed
        elif proximal == 1:
            color = [0, 255, 0]      # Green for Proximal
        elif distal == 1:
            color = [0, 0, 255]      # Blue for Distal
        elif stressed == 1:
            color = [255, 0, 0]      # Red for Stressed
        elif inflamed == 1:
            color = [255,128,0] #orange for inflamed #TODO: test to see if the inflamed labels correctly
        else:
            color = [128, 128, 128]  # Gray for Unclassified
        
        # Apply the color to the tubule instance
        rgb_image[tubules_mask == instance_id] = color
    
    imwrite(output_path, rgb_image)


for DATASET in DATASETS:
    process_tubules(DATASET)

print("Tubule processing complete.")
