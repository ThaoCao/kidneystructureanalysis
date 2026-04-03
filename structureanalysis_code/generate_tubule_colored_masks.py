#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 2025

@author: thaocao
Generate colored tubule masks from reclassified CSV
"""

import os
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_CSV = "/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids_reclassified.csv"
ROOT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Lupus_Nephritis/structural_analysis_instances'
TUBULE_MASK_DIR = os.path.join(ROOT_DIR, "tubules")
OUTPUT_DIR_STATE = os.path.join(ROOT_DIR, "tubule_states_reclassified")
OUTPUT_DIR_TYPE = os.path.join(ROOT_DIR, "tubule_types_reclassified")

NUM_WORKERS = 16

# Color map for visualization
COLOR_MAP = {
    # --- States ---
    'Stressed and Inflamed': (202, 0, 32),    # Deep Red
    'Stressed':              (241, 182, 218), # Pinkish
    'Inflamed':              (253, 184, 99),  # Orange
    'Atrophic':              (150, 100, 100), # Brownish 
    'Healthy':               (184, 225, 134), # Pale Green
    
    # --- Classifications (Fallbacks if not healthy/atrophic) ---
    'Proximal':              (153, 255, 0),   # Bright Green
    'Distal':                (0, 153, 255),   # Blue
    
    # --- Fallback ---
    'Unclassified':          (102, 102, 153)  # Muted Purple/Blue
}

def assign_colors_by_classification(mask, df_sample):
    """
    Generates RGB visualization based on Classification (Proximal/Distal).
    """
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Ensure columns exist
    if 'Proximal' not in df_sample.columns:
        df_sample['Proximal'] = 0
    if 'Distal' not in df_sample.columns:
        df_sample['Distal'] = 0

    
    class_map = df_sample.set_index('TubuleID')[['Proximal', 'Distal']].to_dict('index')
    
    for instance_id, props in class_map.items():
        if props['Proximal']:
            color = COLOR_MAP['Proximal']
        elif props['Distal']:
            color = COLOR_MAP['Distal']
        else:
            color = COLOR_MAP['Unclassified']
        
        rgb_image[mask == instance_id] = color
    
    return rgb_image

def assign_colors_by_state(mask, df_sample):
    """
    Generates RGB visualization based on State (Stressed, Inflamed, Atrophic, Healthy).
    """
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Ensure columns exist
    required_cols = ['Stressed', 'Inflamed', 'Atrophic', 'Healthy']
    for col in required_cols:
        if col not in df_sample.columns:
            df_sample[col] = 0
    
    if 'Stressed and Inflamed' not in df_sample.columns:
        df_sample['Stressed and Inflamed'] = 0
    
    
    class_map = df_sample.set_index('TubuleID')[['Stressed', 'Inflamed', 'Atrophic', 
                                             'Healthy', 'Stressed and Inflamed']].to_dict('index')
    
    for instance_id, props in class_map.items():
        stressed = props['Stressed']
        inflamed = props['Inflamed']
        atrophic = props['Atrophic']
        healthy = props['Healthy']
        stressed_inflamed = props.get('Stressed and Inflamed', 0)
        
        if atrophic:
            color = COLOR_MAP['Atrophic']
        elif stressed_inflamed:
            color = COLOR_MAP['Stressed and Inflamed']
        elif stressed:
            color = COLOR_MAP['Stressed']
        elif inflamed:
            color = COLOR_MAP['Inflamed']
        elif healthy:
            color = COLOR_MAP['Healthy']
        else:
            color = COLOR_MAP['Unclassified']
        
        rgb_image[mask == instance_id] = color
    
    return rgb_image


def assign_colors_to_mask(mask, df_sample):
    """
    Generates the RGB visualization based on the State > Classification hierarchy.
    """
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Ensure all needed columns exist
    required_cols = ['Proximal', 'Distal', 'Stressed', 'Inflamed', 'Atrophic', 'Healthy']
    for col in required_cols:
        if col not in df_sample.columns:
            df_sample[col] = 0
    
    # Add 'Stressed and Inflamed' if it doesn't exist
    if 'Stressed and Inflamed' not in df_sample.columns:
        df_sample['Stressed and Inflamed'] = 0
    
    # Drop duplicate IDs, keeping the first occurrence
    df_sample = df_sample.drop_duplicates(subset='ID', keep='first')
    
    # Create lookup dict for speed
    class_map = df_sample.set_index('ID')[['Proximal', 'Distal', 'Stressed', 'Inflamed', 
                                             'Atrophic', 'Healthy', 'Stressed and Inflamed']].to_dict('index')
    
    for instance_id, props in class_map.items():
        proximal = props['Proximal']
        distal = props['Distal']
        stressed = props['Stressed']
        inflamed = props['Inflamed']
        atrophic = props['Atrophic']
        healthy = props['Healthy']
        stressed_inflamed = props.get('Stressed and Inflamed', 0)
        
        # --- Logic Hierarchy (Pathology > Healthy State > Type) ---
        if atrophic:
            color = COLOR_MAP['Atrophic']
        elif stressed_inflamed:
            color = COLOR_MAP['Stressed and Inflamed']
        elif stressed:
            color = COLOR_MAP['Stressed']
        elif inflamed:
            color = COLOR_MAP['Inflamed']
        elif healthy:
            color = COLOR_MAP['Healthy']
        elif proximal:
            color = COLOR_MAP['Proximal']
        elif distal:
            color = COLOR_MAP['Distal']
        else:
            color = COLOR_MAP['Unclassified']
            
        rgb_image[mask == instance_id] = color
        
    return rgb_image


def process_single_sample(args):
    """
    Process a single sample: load mask, filter CSV data, and generate colored output.
    """
    sample_id, area_id, df = args
    
    sample_area = f"{sample_id}_{area_id}"
    
    # Find matching mask file
    mask_filename = f"{sample_id}_{area_id}.tif"
    mask_path = os.path.join(TUBULE_MASK_DIR, mask_filename)
    
    if not os.path.exists(mask_path):
        return f"[{sample_area}] Error: Mask file not found at {mask_path}"
    
    try:
        # Load mask
        mask = imread(mask_path)
        
        if mask.max() == 0:
            return f"[{sample_area}] Warning: Mask is empty"
        
        # Filter dataframe for this sample
        df_sample = df[(df['Sample'] == sample_id) & (df['Area'] == area_id)].copy()
        
        if df_sample.empty:
            return f"[{sample_area}] Warning: No data found in CSV"
        
        # Generate colored mask
        rgb_mask = assign_colors_by_state(mask, df_sample)
        output_path = os.path.join(OUTPUT_DIR_STATE, f"{sample_area}.tif")
        imwrite(output_path, rgb_mask)
        
        rgb_mask2 = assign_colors_by_classification(mask, df_sample)
        output_path2 = os.path.join(OUTPUT_DIR_TYPE, f"{sample_area}.tif")
        imwrite(output_path2, rgb_mask2)
        
        return f"[{sample_area}] Success: Generated colored mask with {len(df_sample)} tubules"
        
    except Exception as e:
        return f"[{sample_area}] Error: {str(e)}"

def main():
    """
    Main function to process all samples.
    """
    print(f"Reading CSV from: {INPUT_CSV}")
    
    # Check if CSV exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return
    
    # Read CSV
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded CSV with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    if 'Sample' not in df.columns or 'Area' not in df.columns:
        print("Error: CSV must contain 'Sample' and 'Area' columns")
        return
    
    # Create output directory for states
    os.makedirs(OUTPUT_DIR_STATE, exist_ok=True)

    
    # Create output directory for types
    os.makedirs(OUTPUT_DIR_TYPE, exist_ok=True)
    # Get unique sample-area combinations
    unique_samples = df.groupby(['Sample', 'Area']).size().reset_index()[['Sample', 'Area']]
    print(f"\nFound {len(unique_samples)} unique sample-area combinations")
    
    # Prepare arguments for parallel processing
    args_list = [(row['Sample'], row['Area'], df) for _, row in unique_samples.iterrows()]
    
    # Process in parallel
    print(f"\nProcessing with {NUM_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(process_single_sample, args_list))
    
    # Print results
    print("\n" + "=" * 80)
    print("PROCESSING RESULTS")
    print("=" * 80)
    for result in results:
        print(result)
    
    # Summary
    success_count = sum(1 for r in results if 'Success' in r)
    error_count = sum(1 for r in results if 'Error' in r)
    warning_count = sum(1 for r in results if 'Warning' in r)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Warnings: {warning_count}")
    print(f"Errors: {error_count}")

if __name__ == '__main__':
    main()
