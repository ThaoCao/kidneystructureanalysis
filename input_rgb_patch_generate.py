#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:00:21 2025

@author: thaocao
Goal: generate patches from a specific sample area; print out examples from Lupus
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tifffile import imread, imwrite


CONFIG = {
    # Path to the tubule metadata CSV file
    'metadata_file': '/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids.csv',
    
    # Patch size (typically 128 for tubules) #testing 64
    'patch_size': 64,
    
    # Root directory for Lupus Nephritis dataset
    'rootdir': '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets',
    
    # Output directory for patches and metadata
    'output_dir': '/home/thaocao/Palom/data/lupus_nephritis',
    
    # List of specific images to process (set to None to process all images in metadata)
    'img_list': ['A3_Area4.tif', '121422S1_Area2', '112222S4_Area1', '112222S3_Area4'],
    
}



def extract_centered_patch(image, centroid_x, centroid_y, patch_size):
    """
    Extract a patch of fixed size centered on given coordinates.
    Handles boundary cases by padding if necessary.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        centroid_x: X coordinate of center
        centroid_y: Y coordinate of center
        patch_size: Size of output patch
    
    Returns:
        Patch of size (patch_size, patch_size) or (patch_size, patch_size, C)
    """
    half_patch = patch_size // 2
    h, w = image.shape[:2]
    
    # Calculate desired boundaries
    y_min = centroid_y - half_patch
    y_max = centroid_y + half_patch
    x_min = centroid_x - half_patch
    x_max = centroid_x + half_patch
    
    # Check if patch is within image bounds
    if y_min < 0 or y_max > h or x_min < 0 or x_max > w:
        # Need to pad
        if len(image.shape) == 3:
            padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
        
        # Calculate valid regions
        src_y_min = max(0, y_min)
        src_y_max = min(h, y_max)
        src_x_min = max(0, x_min)
        src_x_max = min(w, x_max)
        
        dst_y_min = src_y_min - y_min
        dst_y_max = dst_y_min + (src_y_max - src_y_min)
        dst_x_min = src_x_min - x_min
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        
        padded[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = image[src_y_min:src_y_max, src_x_min:src_x_max]
        return padded
    else:
        return image[y_min:y_max, x_min:x_max]


def get_tubule_mask_patch(tubule_mask, tubule_id, centroid_x, centroid_y, patch_size):
    """
    Extract binary mask for a specific tubule instance.
    
    Args:
        tubule_mask: Full segmentation mask where each tubule has unique ID
        tubule_id: ID of the specific tubule to extract
        centroid_x: X coordinate of tubule centroid
        centroid_y: Y coordinate of tubule centroid
        patch_size: Size of output patch
    
    Returns:
        Binary mask of size (patch_size, patch_size) where 1 = this tubule, 0 = other
    """
    binary_mask = (tubule_mask == tubule_id).astype(np.uint8)
    return extract_centered_patch(binary_mask, centroid_x, centroid_y, patch_size)



def process_lupus_nephritis(config):
    """
    Process Lupus Nephritis tubule dataset.
    Extracts H&E patches, CODEX patches, and tubule masks for all tubule instances.
    
    Args:
        config: Dictionary with configuration parameters
    """
    # Extract config parameters
    metadata_file = config['metadata_file']
    patch_size = config['patch_size']
    rootdir = config['rootdir']
    output_dir = config['output_dir']
    img_list = config['img_list']
    
    # Dataset paths
    dataset = 'Lupus_Nephritis'
    input_dir = 'DAPI_HE_normalized_aligned_rgb_cleaned_ds'
    tubule_codex = 'ds10/rgb_tubules/'
    tubule_seg = 'tubule_segmentations/masks/tubule_cleaned/cleaned'
    
    # Output directories
    output_he_folder = os.path.join(output_dir, f'he_patches_{patch_size}')
    output_codex_folder = os.path.join(output_dir, f'codex_patches_{patch_size}')
    output_masks_folder = os.path.join(output_dir, f'tubule_masks_{patch_size}')
    
    os.makedirs(output_he_folder, exist_ok=True)
    os.makedirs(output_codex_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)
    
    print("\n" + "="*70)
    print("TUBULE PATCH EXTRACTION - LUPUS NEPHRITIS")
    print("="*70)
    print(f"Configuration:")
    print(f"  Metadata file: {metadata_file}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Root directory: {rootdir}")
    print(f"  Output directory: {output_dir}")
    print("="*70)
    
    print(f"\nOutput directories created:")
    print(f"  H&E patches: {output_he_folder}")
    print(f"  CODEX patches: {output_codex_folder}")
    print(f"  Tubule masks: {output_masks_folder}")
    
    # Verify files exist
    if not os.path.exists(metadata_file):
        print(f"\n❌ ERROR: Metadata file not found: {metadata_file}")
        return
    
    if not os.path.exists(rootdir):
        print(f"\n❌ ERROR: Root directory not found: {rootdir}")
        return
    
    # Load metadata
    df = pd.read_csv(metadata_file)
    print(f"\n✓ Loaded metadata with {len(df)} tubule instances")
    print(f"✓ Columns: {df.columns.tolist()}\n")
    
    # Define image list (samples to process)
    if img_list is None:
        # Process all unique samples in metadata
        unique_samples = df['Sample_Area'].unique()
        # Remove 'LuN_' prefix if present to get image names
        img_list = [s.replace('LuN_', '') for s in unique_samples]
        print(f"Processing ALL {len(img_list)} images in metadata\n")
    else:
        print(f"Processing {len(img_list)} specified images: {img_list}\n")
    
    # Paths
    rdir_tubule = os.path.join(rootdir, dataset, tubule_seg)
    rdir_he = os.path.join(rootdir, dataset, input_dir)
    rdir_codex = os.path.join(rootdir, dataset, tubule_codex)
    
    # Get file lists
    tubule_seg_list = os.listdir(rdir_tubule)
    he_img_list = os.listdir(rdir_he)
    tubule_codex_list = os.listdir(rdir_codex)
    
    # Summary statistics
    summary_stats = []
    all_processed_tubules = []
    
    # Process each image
    for img in tqdm(img_list, desc=f"Processing {dataset}"):
        img_name = img.replace('.tif', '')
        img_name_full = 'LuN_' + img_name
        
        # Find matching files
        tubule_seg_img = [f for f in tubule_seg_list if img_name in f]
        tubule_codex_img = [f for f in tubule_codex_list if img_name in f]
        he_img = [f for f in he_img_list if img_name in f]
        
        if len(tubule_seg_img) == 0 or len(he_img) == 0 or len(tubule_codex_img) == 0:
            print(f"\n⚠️  Could not find matching files for {img}")
            print(f"  Segmentation files matching '{img_name}': {len(tubule_seg_img)}")
            print(f"  H&E files matching '{img_name}': {len(he_img)}")
            print(f"  CODEX files matching '{img_name}': {len(tubule_codex_img)}")
            continue
        
        tubule_seg_img = tubule_seg_img[0]
        tubule_codex_img = tubule_codex_img[0]
        he_img = he_img[0]
        
        print(f"\n{'='*70}")
        print(f"Processing: {img_name_full}")
        print(f"{'='*70}")
        print(f"  Segmentation: {tubule_seg_img}")
        print(f"  CODEX: {tubule_codex_img}")
        print(f"  H&E: {he_img}")
        
        # Filter dataframe for this image
        img_df = df[df['Sample_Area'] == img_name_full].copy()
        
        if len(img_df) == 0:
            print(f"  ⚠️  No tubule instances found for {img_name_full}")
            continue
        
        print(f"  ✓ Found {len(img_df)} tubule instances")
        
        # Collect summary statistics
        stats = {
            'Sample': img_name_full,
            'Total_Tubules': len(img_df),
        }
        
        # Add counts by classification type if column exists
        if 'Classification' in img_df.columns:
            classification_counts = img_df['Classification'].value_counts()
            for classification, count in classification_counts.items():
                stats[f'Classification_{classification}'] = count
        
        # Add counts by state if column exists
        if 'State' in img_df.columns:
            state_counts = img_df['State'].value_counts()
            for state, count in state_counts.items():
                stats[f'State_{state}'] = count
        
        summary_stats.append(stats)
        
        # Load images
        print(f"  Loading images...")
        tubule_seg_path = os.path.join(rdir_tubule, tubule_seg_img)
        tubule_mask = imread(tubule_seg_path)
        
        he_image_path = os.path.join(rdir_he, he_img)
        he_image = imread(he_image_path)
        
        codex_image_path = os.path.join(rdir_codex, tubule_codex_img)
        codex_image = imread(codex_image_path)
        
        # Transpose if channels are first (C, H, W) -> (H, W, C)
        if he_image.shape[0] in [3, 4]:
            he_image = np.transpose(he_image, (1, 2, 0))
        
        if codex_image.shape[0] in [3, 4]:
            codex_image = np.transpose(codex_image, (1, 2, 0))
        
        print(f"  Image shapes:")
        print(f"    H&E: {he_image.shape}")
        print(f"    CODEX: {codex_image.shape}")
        print(f"    Mask: {tubule_mask.shape}")
        
        # Process each tubule instance
        print(f"  Extracting patches for {len(img_df)} tubules...")
        patches_saved = 0
        patches_failed = 0
        
        for idx, row in tqdm(img_df.iterrows(), total=len(img_df), 
                            desc=f"  Extracting patches", leave=False):
            try:
                tubule_id = row['TubuleID']
                centroid_x = int(row['Centroid_X'])
                centroid_y = int(row['Centroid_Y'])
                classification = row.get('Classification', 'Unknown')
                state = row.get('State', 'Unknown')
                
                # Extract patches centered on tubule centroid
                he_patch = extract_centered_patch(he_image, centroid_x, centroid_y, patch_size)
                codex_patch = extract_centered_patch(codex_image, centroid_x, centroid_y, patch_size)
                mask_patch = get_tubule_mask_patch(tubule_mask, tubule_id, centroid_x, centroid_y, patch_size)
                
                # Verify patch sizes
                if he_patch.shape[:2] != (patch_size, patch_size):
                    print(f"    ⚠️  H&E patch size mismatch for tubule {tubule_id}: {he_patch.shape}")
                    patches_failed += 1
                    continue
                
                if codex_patch.shape[:2] != (patch_size, patch_size):
                    print(f"    ⚠️  CODEX patch size mismatch for tubule {tubule_id}: {codex_patch.shape}")
                    patches_failed += 1
                    continue
                
                if mask_patch.shape != (patch_size, patch_size):
                    print(f"    ⚠️  Mask patch size mismatch for tubule {tubule_id}: {mask_patch.shape}")
                    patches_failed += 1
                    continue
                
                # Create output filename
                output_name = f"{img_name_full}_{tubule_id}_{classification}_{state}_{centroid_x}_{centroid_y}"
                
                # Save patches
                np.save(os.path.join(output_he_folder, f"{output_name}.npy"), he_patch)
                np.save(os.path.join(output_codex_folder, f"{output_name}.npy"), codex_patch)
                np.save(os.path.join(output_masks_folder, f"{output_name}.npy"), mask_patch)
                
                # Track processed tubule
                tubule_info = {
                    'sample_area': img_name_full,
                    'tubule_id': tubule_id,
                    'classification': classification,
                    'state': state,
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'filename': output_name
                }
                all_processed_tubules.append(tubule_info)
                
                patches_saved += 1
                
            except Exception as e:
                print(f"    ⚠️  Error processing tubule {row.get('TubuleID', 'unknown')}: {e}")
                patches_failed += 1
                continue
        
        print(f"  ✓ Successfully saved {patches_saved} patch sets")
        if patches_failed > 0:
            print(f"  ⚠️  Failed to save {patches_failed} patch sets")
    
    # Save summary statistics
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_path = os.path.join(output_dir, 'tubule_summary_statistics.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print(f"Summary statistics saved to: {summary_path}")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print()
    
    # Save metadata for all processed tubules
    if all_processed_tubules:
        processed_df = pd.DataFrame(all_processed_tubules)
        metadata_path = os.path.join(output_dir, 'processed_tubule_metadata.csv')
        processed_df.to_csv(metadata_path, index=False)
        print(f"\n✓ Processed tubule metadata saved to: {metadata_path}")
        print(f"✓ Total tubules processed: {len(processed_df)}")
        
        # Print distribution
        if 'classification' in processed_df.columns:
            print(f"\nClassification distribution:")
            print(processed_df['classification'].value_counts().to_string())
        
        if 'state' in processed_df.columns:
            print(f"\nState distribution:")
            print(processed_df['state'].value_counts().to_string())
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\nStarting extraction with configuration:")
    print(f"  Metadata: {CONFIG['metadata_file']}")
    print(f"  Patch size: {CONFIG['patch_size']}")
    print(f"  Output: {CONFIG['output_dir']}")
    if CONFIG['img_list']:
        print(f"  Images: {len(CONFIG['img_list'])} specified")
    else:
        print(f"  Images: All in metadata")
    print("="*70)
    
    process_lupus_nephritis(CONFIG)