#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 2025

@author: thaocao
Reclassify tubules from existing CSV data
"""

import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops

# Configuration
INPUT_CSV = '/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids.csv'
OUTPUT_CSV = '/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids_reclassified2.csv'
# Markers used in classification
MARKERS = ["CD45", "Claudin1", "MUC1", "CD10"]

# Classification Constants
INTENSITY_THRESHOLD = 20
FREQUENCY_THRESHOLD = 10

# Columns to reinitialize to 0
COLUMNS_TO_RESET = ['Proximal', 'Distal', 'Stressed', 'Inflamed', 'Atrophic', 'Healthy', 'Unhealthy','Unclassified']

# Columns to reset string values to 'n/a'
STRING_COLUMNS_TO_RESET = ['Classification', 'State']

# Column to add
COLUMNS_TO_ADD = ['Stressed and Inflamed', 'Circularity']

# Root directory for mask files
ROOT_DIR = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Lupus_Nephritis/structural_analysis_instances'
TUBULES_SEG = 'tubules'

def calculate_circularity_for_all(df):
    """
    Calculate circularity for all tubules in the dataframe by reading mask files.
    Updates the 'Circularity' column in place.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing tubule data with Dataset, Sample, Area, and TubuleID columns
        
    Returns:
    --------
    pandas.DataFrame
        Updated DataFrame with Circularity values
    """
    # Initialize Circularity column if it doesn't exist
    if 'Circularity' not in df.columns:
        df['Circularity'] = 0.0
    
    # Get unique combinations of Dataset, Sample, Area
    unique_combinations = df[['Dataset', 'Sample', 'Area']].drop_duplicates()
    
    print(f"\nCalculating circularity for {len(unique_combinations)} unique sample areas...")
    
    for _, row in unique_combinations.iterrows():
        dataset = row['Dataset']
        sample = row['Sample']
        area = row['Area']
        
        # Construct the file path
        tubules_seg_dir = os.path.join(ROOT_DIR, TUBULES_SEG)
        sample_area = f"{sample}_{area}"
        filename = f"{sample_area}.tif"
        file_path = os.path.join(tubules_seg_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f'Processing {sample_area} in {dataset}')
        
        try:
            tubule_mask = imread(file_path)
            
            # Get unique instance IDs (excluding background 0)
            instance_ids = np.unique(tubule_mask)
            instance_ids = instance_ids[instance_ids != 0]
            
            # Get TubuleIDs for this sample/area in the DataFrame
            mask = (
                (df['Dataset'] == dataset) &
                (df['Sample'] == sample) &
                (df['Area'] == area)
            )
            
            # Process each instance
            for instance_id in instance_ids:
                tubule_id = int(instance_id)
                
                # Create binary mask for this instance
                instance_mask = (tubule_mask == instance_id).astype(int)
                
                # Calculate properties
                props = regionprops(instance_mask)
                
                if props:
                    props = props[0]
                    area_px = props.area
                    perimeter = props.perimeter
                    
                    # Calculate circularity: 4π × area / perimeter²
                    if perimeter > 0:
                        circularity = (4 * np.pi * area_px) / (perimeter ** 2)
                        circularity = min(circularity, 1.0)  # Cap at 1.0
                    else:
                        circularity = 0.0
                    
                    # Update DataFrame
                    tubule_mask_df = mask & (df['TubuleID'] == tubule_id)
                    if tubule_mask_df.any():
                        df.loc[tubule_mask_df, 'Circularity'] = circularity
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return df

def classify_tubule(row):
    """
    Complete classification logic for a single tubule.
    Determines both Classification (type) and State (pathology).
    """
    # Reset all classification columns to 0
    for col in COLUMNS_TO_RESET:
        row[col] = 0
    
    # Ensure 'Stressed and Inflamed' column exists
    if 'Stressed and Inflamed' not in row.index:
        row['Stressed and Inflamed'] = 0
    
    # Get marker values
    cd10_p95 = row.get('CD10', 0)
    muc1_p95 = row.get('MUC1', 0)
    claudin1_p95 = row.get('Claudin1', 0)
    claudin1_freq = row.get('Claudin1_freq', 0)
    cd45_p95 = row.get('CD45', 0)
    cd45_freq = row.get('CD45_freq', 0)
    
    # ===== STEP 1: TYPE CLASSIFICATION =====
    # Proximal (CD10 >= 20)
    if cd10_p95 >= INTENSITY_THRESHOLD:
        row['Proximal'] = 1
    
    # Distal (MUC1 >= 20)
    if muc1_p95 >= INTENSITY_THRESHOLD:
        row['Distal'] = 1
    
    # Ratio-based classification for edge cases (neither passed threshold)
    if row['Proximal'] == 0 and row['Distal'] == 0 and muc1_p95 > 0:
        ratio = cd10_p95 / muc1_p95
        if ratio < 0.95:
            row['Distal'] = 1
        elif ratio > 1.05:
            row['Proximal'] = 1
    
    # Mark as Unclassified if no type assigned
    if row['Proximal'] == 0 and row['Distal'] == 0:
        row['Unclassified'] = 1
    
    # Determine Classification string
    if row['Proximal'] == 1 and row['Distal'] == 1:
        row['Classification'] = 'Unclassified'
    elif row['Proximal'] == 1:
        row['Classification'] = 'Proximal'
    elif row['Distal'] == 1:
        row['Classification'] = 'Distal'
    else:
        row['Classification'] = 'Unclassified'
    
    # ===== STEP 2: PATHOLOGY CLASSIFICATION =====
    # Stressed (Claudin1 >= 20 and freq > 10)
    if claudin1_p95 >= INTENSITY_THRESHOLD and claudin1_freq > FREQUENCY_THRESHOLD:
        row['Stressed'] = 1
    
    # Inflamed (CD45 >= 40 and freq > 10)
    if cd45_p95 >= 40 and cd45_freq > FREQUENCY_THRESHOLD:
        row['Inflamed'] = 1
    
    # Stressed and Inflamed (both conditions met)
    if row['Stressed'] == 1 and row['Inflamed'] == 1:
        row['Stressed'] = 0
        row['Inflamed'] = 0
        row['Stressed and Inflamed'] = 1
    
    # ===== STEP 3: ATROPHIC STATE =====
    # Only for Unclassified type, with size/circularity criteria, no other pathology
    if (row['Unclassified'] == 1 and 
        row['Stressed'] == 0 and 
        row['Inflamed'] == 0 and 
        row['Stressed and Inflamed'] == 0 and 
        row.get('size', 0) > 69 and 
        row.get('Circularity', 0) > 0.892805):
        row['Atrophic'] = 1
    
    # ===== STEP 4: HEALTHY STATE =====
    # No pathological markers (can be any type)
    if (row['Stressed'] == 0 and 
        row['Inflamed'] == 0 and 
        row['Stressed and Inflamed'] == 0 and 
        row['Atrophic'] == 0):
        row['Healthy'] = 1
    
    # ===== DETERMINE STATE STRING =====
    # Priority: Atrophic > Stressed+Inflamed > Stressed > Inflamed > Healthy
    if row['Atrophic'] == 1:
        row['State'] = 'Atrophic'
    elif row['Stressed and Inflamed'] == 1:
        row['State'] = 'Stressed and Inflamed'
    elif row['Stressed'] == 1:
        row['State'] = 'Stressed'
    elif row['Inflamed'] == 1:
        row['State'] = 'Inflamed'
    elif row['Healthy'] == 1:
        row['State'] = 'Healthy'
    else:
        row['State'] = 'Healthy'
    
    return row

def main():
    """
    Main function to read CSV, remove duplicates, reclassify, and save.
    """
    print(f"Reading CSV from: {INPUT_CSV}")
    
    # Read the CSV
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV not found at {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print(f"Initial shape: {df.shape}")
    print(f"\nAll columns in the dataframe:")
    print(df.columns.tolist())
    
    # Remove duplicate rows
    initial_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    print(f"Shape after removing duplicates: {df.shape}")

    # Calculate circularity for all tubules
    print("\nCalculating circularity for tubules...")
    df = calculate_circularity_for_all(df)
    print(f"Circularity calculation complete")
    print(f"\nCircularity statistics:")
    print(f"  Min: {df['Circularity'].min():.6f}")
    print(f"  Max: {df['Circularity'].max():.6f}")
    print(f"  Mean: {df['Circularity'].mean():.6f}")
    print(f"  Median: {df['Circularity'].median():.6f}")
    
    # Ensure all classification columns exist
    for col in COLUMNS_TO_RESET:
        if col not in df.columns:
            df[col] = 0
    
    if 'Stressed and Inflamed' not in df.columns:
        df['Stressed and Inflamed'] = 0
    
    # Apply reclassification to each row
    print("Reclassifying tubules...")
    df = df.apply(classify_tubule, axis=1)
    
    # Debug: Check marker value ranges
    print("\n" + "=" * 50)
    print("MARKER VALUE STATISTICS")
    print("=" * 50)
    for marker in MARKERS:
        p95_col = f'{marker}_95th'
        freq_col = f'{marker}_freq'
        if p95_col in df.columns:
            print(f"\n{marker}_95th:")
            print(f"  Min: {df[p95_col].min():.2f}, Max: {df[p95_col].max():.2f}, Mean: {df[p95_col].mean():.2f}")
            print(f"  >= {INTENSITY_THRESHOLD}: {(df[p95_col] >= INTENSITY_THRESHOLD).sum()} rows")
        if freq_col in df.columns:
            print(f"{marker}_freq:")
            print(f"  Min: {df[freq_col].min():.2f}, Max: {df[freq_col].max():.2f}, Mean: {df[freq_col].mean():.2f}")
            print(f"  > {FREQUENCY_THRESHOLD}: {(df[freq_col] > FREQUENCY_THRESHOLD).sum()} rows")
    
    # Print classification summary
    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print("\nState Distribution:")
    print(df['State'].value_counts())
    print("\nClassification (Type) Distribution:")
    print(df['Classification'].value_counts())
    
    # Print flag counts
    print("\n" + "=" * 50)
    print("FLAG COUNTS")
    print("=" * 50)
    for col in COLUMNS_TO_RESET:
        if col in df.columns:
            count = (df[col] == 1).sum()
            print(f"{col}: {count} ({count/len(df)*100:.2f}%)")
    if 'Stressed and Inflamed' in df.columns:
        count = (df['Stressed and Inflamed'] == 1).sum()
        print(f"Stressed and Inflamed: {count} ({count/len(df)*100:.2f}%)")
    
    # Save the reclassified dataframe
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nReclassified CSV saved to: {OUTPUT_CSV}")
    print(f"Final shape: {df.shape}")

if __name__ == '__main__':
    main()

    def qc_unique_percentages():
        """
        Quality control function to check percentage of unique values in Classification and State columns.
        """
        if not os.path.exists(OUTPUT_CSV):
            print(f"Error: Output CSV not found at {OUTPUT_CSV}")
            return
        
        df = pd.read_csv(OUTPUT_CSV)
        
        print("\n" + "=" * 50)
        print("QUALITY CONTROL - UNIQUE VALUE PERCENTAGES")
        print("=" * 50)
        
        for col in ['Classification', 'State']:
            if col in df.columns:
                value_counts = df[col].value_counts()
                total = len(df)
                
                print(f"\n{col}:")
                for value, count in value_counts.items():
                    percentage = (count / total) * 100
                    print(f"  {value}: {count} ({percentage:.2f}%)")
            else:
                print(f"\n{col}: Column not found")
        
        print("=" * 50)

    # Call QC function after main execution
    if __name__ == '__main__':
        main()
        qc_unique_percentages()