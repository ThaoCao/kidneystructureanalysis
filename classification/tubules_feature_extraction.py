#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:41:47 2025

@author: thaocao
Goal: extract and save single tubule instances' features
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from scipy.stats import mannwhitneyu
from scipy.ndimage import distance_transform_edt
import seaborn as sns

ROOT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/'
DATASETS = ['Antibody_Mediated_Rejection', 'Lupus_Nephritis', 'Renal_Allograft','Normal_Kidney']
TUBULES_SEG = 'structural_analysis_instances/vessels/'
MARKER_DIR = 'ds10'
MARKERS = ['CD45', 'Claudin1', 'MUC1', 'CD10']
OUTPUT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/structure_analysis/vessels_px_all_95th.csv'

master_dataframes = [] #a list storing all dataset-based dataframes

def characterize_tubules(DATASET):
    """
    Calculate and save the characteristics of every tubule instance in a csv {sample_area}.csv 
    """
    tubules_seg_dir = os.path.join(ROOT_DIR, DATASET, TUBULES_SEG)
    
    for filename in os.listdir(tubules_seg_dir):
        if not filename.endswith('.tif'):
            continue
        
        filename_path = os.path.join(tubules_seg_dir, filename)
        tubules_mask = imread(filename_path)
        sample_area = os.path.splitext(filename)[0]
        
        # Split sample_area into sample and area
        parts = sample_area.split('_')
        if len(parts) < 2:
            print(f"Warning: Could not parse sample and area from filename {filename}. Skipping.")
            continue
        sample = '_'.join(parts[:-1])
        area = parts[-1]
        
        # Initialize a list to store data for this sample
        sample_data = []
        
        # Iterate over each marker and calculate the pixel percentiles and frequency
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
            
            # Iterate over each marker and calculate the pixel percentiles and frequency
            for instance_id in instance_ids:
                instance_mask = (tubules_mask == instance_id)
                size = np.sum(instance_mask)
                
                # Initialize a dictionary to store data for this instance
                instance_data = {
                    "Dataset": DATASET,
                    "Sample": sample,
                    "Area": area,
                    "VesselID": instance_id,
                    "size": size,
                    "CD45": np.nan,
                }
                
                # Iterate over each marker
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
                    
                    marker_pixels = marker_image[instance_mask]
                    
                    # Calculate 95th percentile for the current marker
                    percentile = np.percentile(marker_pixels, 95) if len(marker_pixels) > 0 else np.nan
                    
                    # Update the instance data with the percentile
                    instance_data[MARKER] = percentile
                    
                    # Calculate frequency of the most common pixel value for this marker
                    values, counts = np.unique(marker_pixels, return_counts=True)
                    freq = counts.max() if len(counts) > 0 else 0  # Frequency of the most common intensity
                    instance_data[f"{MARKER}_freq"] = freq
                
                # Append data to the sample list
                sample_data.append(instance_data)
        
        # Convert sample data to DataFrame and append to master list
        if sample_data:
            sample_df = pd.DataFrame(sample_data)
            master_dataframes.append(sample_df)


dataset_dataframes = []
for DATASET in DATASETS:
    characterize_tubules(DATASET)
    ataset_df = pd.concat(master_dataframes, ignore_index=True)
    # Concatenate all DataFrames for this dataset
    if master_dataframes:
        dataset_df = pd.concat(master_dataframes, ignore_index=True)
        
        # Plot histograms for each column except ['Dataset', 'Sample', 'Area', 'TubuleID']
        columns_to_plot = [col for col in dataset_df.columns if col not in ['Dataset', 'Sample', 'Area', 'TubuleID']]
        
        fig, axes = plt.subplots(nrows=len(columns_to_plot), ncols=1, figsize=(8, 6*len(columns_to_plot)))
        
        if len(columns_to_plot) == 1:
            axes = [axes]
        
        for ax, col in zip(axes, columns_to_plot):
            dataset_df[col].plot.hist(ax=ax, bins=255)
            ax.set_title(f"Histogram of {col} for {DATASET}")
            ax.set_xlabel(col)
            plt.yscale('log')
            ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()
    dataset_dataframes.append(dataset_df)

dataset_df_concat     = pd.concat(dataset_dataframes, ignore_index=True)
dataset_df_concat.to_csv('vessels_instances.csv', index=False)

if dataset_dataframes:
    master_df = pd.concat(dataset_dataframes, ignore_index=True)
    
    # Save the DataFrame to the specified output directory
    master_df.to_csv(OUTPUT_DIR, index=False)
else:
    print("No data collected. Output CSV will be empty.")import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tifffile import imread, imwrite
    from scipy.stats import mannwhitneyu
    from scipy.ndimage import distance_transform_edt
    import seaborn as sns

    ROOT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/'
    DATASETS = ['Antibody_Mediated_Rejection', 'Lupus_Nephritis', 'Renal_Allograft','Normal_Kidney']
    TUBULES_SEG = 'structural_analysis_instances/vessels/'
    MARKER_DIR = 'ds10'
    MARKERS = ['CD45']
    OUTPUT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/structure_analysis/vessels_px_all_95th.csv'

    master_dataframes = [] #a list storing all dataset-based dataframes

    def characterize_tubules(DATASET):
        """
        Calculate and save the characteristics of every tubule instance in a csv {sample_area}.csv 
        """
        tubules_seg_dir = os.path.join(ROOT_DIR, DATASET, TUBULES_SEG)
        
        for filename in os.listdir(tubules_seg_dir):
            if not filename.endswith('.tif'):
                continue
            
            filename_path = os.path.join(tubules_seg_dir, filename)
            tubules_mask = imread(filename_path)
            sample_area = os.path.splitext(filename)[0]
            
            # Split sample_area into sample and area
            parts = sample_area.split('_')
            if len(parts) < 2:
                print(f"Warning: Could not parse sample and area from filename {filename}. Skipping.")
                continue
            sample = '_'.join(parts[:-1])
            area = parts[-1]
            
            # Initialize a list to store data for this sample
            sample_data = []
            
            # Iterate over each marker and calculate the pixel percentiles and frequency
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
                
                # Iterate over each marker and calculate the pixel percentiles and frequency
                for instance_id in instance_ids:
                    instance_mask = (tubules_mask == instance_id)
                    size = np.sum(instance_mask)
                    
                    # Initialize a dictionary to store data for this instance
                    instance_data = {
                        "Dataset": DATASET,
                        "Sample": sample,
                        "Area": area,
                        "VesselID": instance_id,
                        "size": size,
                        "CD45": np.nan,
                    }
                    
                    # Iterate over each marker
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
                        
                        marker_pixels = marker_image[instance_mask]
                        
                        # Calculate 95th percentile for the current marker
                        percentile = np.percentile(marker_pixels, 95) if len(marker_pixels) > 0 else np.nan
                        
                        # Update the instance data with the percentile
                        instance_data[MARKER] = percentile
                        
                        # Calculate frequency of the most common pixel value for this marker
                        values, counts = np.unique(marker_pixels, return_counts=True)
                        freq = counts.max() if len(counts) > 0 else 0  # Frequency of the most common intensity
                        instance_data[f"{MARKER}_freq"] = freq
                    
                    # Append data to the sample list
                    sample_data.append(instance_data)
            
            # Convert sample data to DataFrame and append to master list
            if sample_data:
                sample_df = pd.DataFrame(sample_data)
                master_dataframes.append(sample_df)


    dataset_dataframes = []
    for DATASET in DATASETS:
        characterize_tubules(DATASET)
        ataset_df = pd.concat(master_dataframes, ignore_index=True)
        # Concatenate all DataFrames for this dataset
        if master_dataframes:
            dataset_df = pd.concat(master_dataframes, ignore_index=True)
            
            # Plot histograms for each column except ['Dataset', 'Sample', 'Area', 'TubuleID']
            columns_to_plot = [col for col in dataset_df.columns if col not in ['Dataset', 'Sample', 'Area', 'TubuleID']]
            
            fig, axes = plt.subplots(nrows=len(columns_to_plot), ncols=1, figsize=(8, 6*len(columns_to_plot)))
            
            if len(columns_to_plot) == 1:
                axes = [axes]
            
            for ax, col in zip(axes, columns_to_plot):
                dataset_df[col].plot.hist(ax=ax, bins=255)
                ax.set_title(f"Histogram of {col} for {DATASET}")
                ax.set_xlabel(col)
                plt.yscale('log')
                ax.set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.show()
        dataset_dataframes.append(dataset_df)

    dataset_df_concat     = pd.concat(dataset_dataframes, ignore_index=True)
    dataset_df_concat.to_csv('vessels_instances.csv', index=False)

    if dataset_dataframes:
        master_df = pd.concat(dataset_dataframes, ignore_index=True)
        
        # Save the DataFrame to the specified output directory
        master_df.to_csv(OUTPUT_DIR, index=False)
    else:
        print("No data collected. Output CSV will be empty.")