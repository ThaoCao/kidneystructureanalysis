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
    #TODO: change this to your CSV metadata file path where the tubule classifications labels are stored
    'metadata_file': '/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids.csv',
    
    # Patch size (typically 128 for tubules) #testing 64
    'patch_size': 64,
    
    # Root directory
    'rootdir': '/project/mclark/SCAMPI_datasets/',
    
    # Output directory for patches and metadata 
    #TODO: change this to your output directory for the test set (e.g., /home/thaocao/Palom/data/testset_02042026)
    'output_dir': '/home/thaocao/Palom/data/testset_02042026',

    # Datasets
    'datasets': ['Lupus_Nephritis_60ch'],
    
    # List of specific images to process (set to None to process all images in metadata)
    #TODO: change this to specify the composites you want to process
    'img_list': ['03232023S3_Area2', '04182023S2_Area1'],
    
    # CODEX marker channels to extract (ORDER MATTERS!)
    'markers': ['CD10_', 'Claudin1', 'MUC1', 'CD45'],
    
    # Directory suffix for marker folders (e.g., 'ds10' for downsampled images)
    'marker_dir_suffix': 'ds10',
}
import os

# Create directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)


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


def verify_marker_directories(rootdir, dataset, marker_dirs, markers):
    """
    Verify all marker directories exist and contain files before extraction.
    
    Returns:
        Tuple of (all_exist, available_markers, missing_markers)
    """
    print("\n" + "="*70)
    print("MARKER DIRECTORY VERIFICATION")
    print("="*70)
    
    available_markers = []
    missing_markers = []
    
    for marker in markers:
        if marker not in marker_dirs:
            print(f"  ❌ {marker:15s}: Not in marker_dirs configuration")
            missing_markers.append(marker)
            continue
        
        marker_dir = marker_dirs[marker]
        full_path = os.path.join(rootdir, dataset, marker_dir)
        
        if os.path.exists(full_path):
            files = [f for f in os.listdir(full_path) if f.endswith(('.tif', '.tiff', '.TIF', '.TIFF'))]
            n_files = len(files)
            if n_files > 0:
                print(f"  ✓ {marker:15s}: {full_path} ({n_files} files)")
                available_markers.append(marker)
            else:
                print(f"  ⚠️  {marker:15s}: {full_path} (EXISTS but NO .tif FILES)")
                missing_markers.append(marker)
        else:
            print(f"  ❌ {marker:15s}: {full_path} (DIRECTORY NOT FOUND)")
            missing_markers.append(marker)
    
    print("="*70)
    
    return len(missing_markers) == 0, available_markers, missing_markers


def load_multichannel_codex(rootdir, dataset, marker_dirs, img_name, markers, patch_reference_shape=None):
    """
    Load all CODEX marker channels and stack them into a single multi-channel image.
    Creates zero-filled channels for missing markers to ensure consistent output shape.
    
    Args:
        rootdir: Root directory path
        dataset: Dataset name (e.g., 'Lupus_Nephritis')
        marker_dirs: Dictionary mapping marker names to directory paths
        img_name: Image name to load
        markers: List of marker names in desired order (MUST always be 4 markers)
        patch_reference_shape: Reference (H, W) shape, if available
    
    Returns:
        Tuple of (Multi-channel CODEX image of shape (H, W, 4), list of status per marker)
    """
    channel_images = []
    marker_status = []  # Track which markers loaded successfully
    reference_shape = None
    
    print(f"    Loading CODEX markers for {img_name}...")
    
    for marker_idx, marker in enumerate(markers):
        marker_loaded = False
        marker_img = None
        
        if marker not in marker_dirs:
            print(f"      ⚠️  {marker}: Not in configuration")
        else:
            marker_dir = marker_dirs[marker]
            marker_path = os.path.join(rootdir, dataset, marker_dir)
            
            # Find matching file for this marker
            if os.path.exists(marker_path):
                marker_files = [f for f in os.listdir(marker_path) 
                               if img_name in f and f.endswith(('.tif', '.tiff', '.TIF', '.TIFF'))]
                
                if len(marker_files) > 0:
                    marker_file = marker_files[0]
                    marker_img_path = os.path.join(marker_path, marker_file)
                    
                    try:
                        marker_img = imread(marker_img_path)
                        
                        # Handle different image formats
                        if marker_img.ndim == 3:
                            # If multi-channel, take first channel
                            if marker_img.shape[0] in [3, 4]:
                                marker_img = marker_img[0]  # Channels first (C, H, W)
                            elif marker_img.shape[-1] in [3, 4]:
                                marker_img = marker_img[:, :, 0]  # Channels last (H, W, C)
                            else:
                                marker_img = marker_img[:, :, 0]
                        
                        # Ensure 2D
                        if marker_img.ndim != 2:
                            marker_img = marker_img.reshape(marker_img.shape[:2])
                        
                        # Check if mostly zeros
                        non_zero_count = np.count_nonzero(marker_img)
                        total_pixels = marker_img.size
                        non_zero_pct = 100.0 * non_zero_count / total_pixels if total_pixels > 0 else 0
                        
                        # Store shape as reference
                        if reference_shape is None:
                            reference_shape = marker_img.shape
                        
                        marker_loaded = True
                        
                        if non_zero_pct < 0.1:
                            print(f"      ℹ️  {marker}: Loaded (shape={marker_img.shape}, {non_zero_pct:.4f}% non-zero - mostly empty)")
                        else:
                            print(f"      ✓ {marker}: Loaded (shape={marker_img.shape}, {non_zero_pct:.2f}% non-zero)")
                        
                    except Exception as e:
                        print(f"      ❌ {marker}: Error loading: {e}")
                else:
                    print(f"      ⚠️  {marker}: No matching file found")
            else:
                print(f"      ❌ {marker}: Directory does not exist")
        
        # If marker wasn't loaded, create zero channel
        if not marker_loaded or marker_img is None:
            # Determine shape for zero channel
            if reference_shape is not None:
                zero_shape = reference_shape
            elif len(channel_images) > 0:
                zero_shape = channel_images[0].shape
            elif patch_reference_shape is not None:
                zero_shape = patch_reference_shape
            else:
                # Default fallback
                raise ValueError(f"Cannot determine shape for zero channel {marker}: no reference available")
            
            marker_img = np.zeros(zero_shape, dtype=np.uint16)
            marker_status.append(f"{marker}_ZERO")
            print(f"      ⚠️  {marker}: Created ZERO channel (shape={zero_shape})")
            
            if reference_shape is None:
                reference_shape = zero_shape
        else:
            marker_status.append(marker)
        
        # Ensure marker_img has correct shape
        if marker_img.shape != reference_shape:
            print(f"      ⚠️  {marker}: Shape mismatch {marker_img.shape} vs {reference_shape}, creating zero channel")
            marker_img = np.zeros(reference_shape, dtype=np.uint16)
            if marker_status[-1] != f"{marker}_ZERO":
                marker_status[-1] = f"{marker}_ZERO"
        
        channel_images.append(marker_img)
    
    # Verify we have exactly 4 channels
    if len(channel_images) != 4:
        raise ValueError(f"Expected 4 marker channels, got {len(channel_images)}")
    
    # Stack all channels (H, W, 4)
    multichannel_codex = np.stack(channel_images, axis=-1)
    
    print(f"    ✓ Multi-channel CODEX created: shape={multichannel_codex.shape}, dtype={multichannel_codex.dtype}")
    print(f"    ✓ Channel status: {marker_status}")
    
    # Verify final shape
    assert multichannel_codex.shape[2] == 4, f"Final CODEX must have 4 channels, got {multichannel_codex.shape}"
    
    return multichannel_codex, marker_status


def process_balanced_datasets(config):
        """
        Process specified dataset and images.
        Extracts H&E patches, multi-channel CODEX patches (always 4 channels), and tubule masks.
        
        Args:
            config: Dictionary with configuration parameters
        """
        # Extract config parameters
        metadata_file = config['metadata_file']
        patch_size = config['patch_size']
        rootdir = config['rootdir']
        output_dir = config['output_dir']
        markers = config['markers']
        marker_dir_suffix = config['marker_dir_suffix']
        datasets = config['datasets']
        img_list = config['img_list']
        
        # Verify we have exactly 4 markers
        if len(markers) != 4:
            raise ValueError(f"CONFIG must specify exactly 4 markers, got {len(markers)}: {markers}")
        
        # Load metadata and create balanced dataset
        if not os.path.exists(metadata_file):
            print(f"❌ ERROR: Metadata file not found: {metadata_file}")
            return
        
        df = pd.read_csv(metadata_file)
        print(f"✓ Loaded metadata with {len(df)} tubule instances\n")
        
        # Add dataset column based on Sample_Area prefix
        df['Dataset'] = df['Sample_Area'].apply(
            lambda x: 'Lupus_Nephritis' if x.startswith('LuN_') else 'Renal_Allograft'
        )
        
        # Filter to only specified dataset
        if len(datasets) == 1:
            dataset = datasets[0]
            df = df[df['Dataset'] == dataset].copy()
            print(f"Filtered to dataset: {dataset}")
        
        # Filter to only specified images if img_list is provided
        if img_list is not None and len(img_list) > 0:
            # Add prefix to img_list based on dataset
            if dataset == 'Lupus_Nephritis':
                prefixed_imgs = [f'LuN_{img}' for img in img_list]
            else:
                prefixed_imgs = [f'MR_{img}' for img in img_list]
            
            df = df[df['Sample_Area'].isin(prefixed_imgs)].copy()
            print(f"Filtered to {len(img_list)} specified images: {img_list}")
        
        print(f"\nTotal tubules to process: {len(df)}")
        print(f"Unique images: {df['Sample_Area'].nunique()}")
        
        # Output directories
        output_he_folder = os.path.join(output_dir, f'he_patches_{patch_size}')
        output_codex_folder = os.path.join(output_dir, f'codex_patches_{patch_size}')
        output_masks_folder = os.path.join(output_dir, f'tubule_masks_{patch_size}')
        
        os.makedirs(output_he_folder, exist_ok=True)
        os.makedirs(output_codex_folder, exist_ok=True)
        os.makedirs(output_masks_folder, exist_ok=True)
        
        # Process the dataset
        all_processed_tubules = []
        global_channel_stats = {
            'total_patches': 0,
            'zero_channels': {marker: 0 for marker in markers}
        }
        
        print("\n" + "="*70)
        print(f"PROCESSING {dataset}")
        print("="*70)
        
        # Dataset-specific paths
        if dataset == 'Lupus_Nephritis':
            input_dir = 'DAPI_HE_normalized_aligned_rgb_cleaned_ds'
            tubule_seg = os.path.join('tubule_segmentations', 'masks', 'tubule_cleaned', 'cleaned')
        else:  # Renal_Allograft
            input_dir = 'DAPI_HE_normalized_aligned_rgb_cleaned_ds'
            tubule_seg = os.path.join('tubule_segmentations', 'masks', 'tubule_cleaned', 'cleaned')
        
        # Build marker directory mapping
        marker_dirs = {}
        for marker in markers:
            marker_dirs[marker] = os.path.join(marker_dir_suffix, marker)
        
        # Get unique images from filtered dataframe
        unique_samples = df['Sample_Area'].unique()
        
        if dataset == 'Lupus_Nephritis':
            process_img_list = [s.replace('LuN_', '') for s in unique_samples]
        else:
            process_img_list = [s.replace('MR_', '') for s in unique_samples]
        
        print(f"Processing {len(process_img_list)} images\n")
            
        rdir_tubule = os.path.join(rootdir, dataset, tubule_seg)
        rdir_he = os.path.join(rootdir, dataset, input_dir)
        
        tubule_seg_list = os.listdir(rdir_tubule)
        he_img_list = os.listdir(rdir_he)
        
        # Process each image
        for img in tqdm(process_img_list, desc=f"Processing {dataset}"):
            img_name = img.replace('.tif', '')
            if dataset == 'Lupus_Nephritis':
                img_name_full = 'LuN_' + img_name
            else:
                img_name_full = 'MR_' + img_name
            
            # Get tubules for this image from filtered dataframe
            img_df = df[df['Sample_Area'] == img_name_full].copy()
                
            if len(img_df) == 0:
                continue
            
            # Find matching files
            tubule_seg_img = [f for f in tubule_seg_list if img_name in f]
            he_img = [f for f in he_img_list if img_name in f]
            
            if len(tubule_seg_img) == 0 or len(he_img) == 0:
                print(f"\n⚠️  Skipping {img}: Missing files")
                continue
            
            tubule_seg_img = tubule_seg_img[0]
            he_img = he_img[0]
            
            print(f"\nProcessing: {img_name_full} ({len(img_df)} tubules)")
            
            # Load images
            tubule_seg_path = os.path.join(rdir_tubule, tubule_seg_img)
            tubule_mask = imread(tubule_seg_path)
            
            he_image_path = os.path.join(rdir_he, he_img)
            he_image = imread(he_image_path)
            
            if he_image.shape[0] in [3, 4]:
                he_image = np.transpose(he_image, (1, 2, 0))
            
            try:
                codex_image, marker_status = load_multichannel_codex(
                    rootdir, dataset, marker_dirs, img_name, markers,
                    patch_reference_shape=he_image.shape[:2]
                )
            except Exception as e:
                print(f"  ❌ FATAL: Could not load CODEX for {img_name}: {e}")
                continue
            
            # Process each tubule from filtered dataset
            for idx, row in img_df.iterrows():
                try:
                    tubule_id = row['TubuleID']
                    centroid_x = int(row['Centroid_X'])
                    centroid_y = int(row['Centroid_Y'])
                    classification = row.get('Classification', 'Unknown')
                    state = row.get('State', 'Unknown')
                    
                    he_patch = extract_centered_patch(he_image, centroid_x, centroid_y, patch_size)
                    codex_patch = extract_centered_patch(codex_image, centroid_x, centroid_y, patch_size)
                    mask_patch = get_tubule_mask_patch(tubule_mask, tubule_id, centroid_x, centroid_y, patch_size)
                    
                    if he_patch.shape[:2] != (patch_size, patch_size):
                        continue
                    if codex_patch.shape != (patch_size, patch_size, 4):
                        continue
                    if mask_patch.shape != (patch_size, patch_size):
                        continue
                    
                    output_name = f"{img_name_full}_{tubule_id}_{classification}_{state}_{centroid_x}_{centroid_y}"
                    
                    np.save(os.path.join(output_he_folder, f"{output_name}.npy"), he_patch)
                    np.save(os.path.join(output_codex_folder, f"{output_name}.npy"), codex_patch)
                    np.save(os.path.join(output_masks_folder, f"{output_name}.npy"), mask_patch)
                    
                    global_channel_stats['total_patches'] += 1
                    for i, status in enumerate(marker_status):
                        if '_ZERO' in status:
                            global_channel_stats['zero_channels'][markers[i]] += 1
                    
                    tubule_info = {
                        'dataset': dataset,
                        'sample_area': img_name_full,
                        'tubule_id': tubule_id,
                        'classification': classification,
                        'state': state,
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'filename': output_name,
                    }
                    all_processed_tubules.append(tubule_info)
                    
                except Exception as e:
                    continue
        
        # Save final metadata
        if all_processed_tubules:
            processed_df = pd.DataFrame(all_processed_tubules)
            metadata_path = os.path.join(output_dir, 'processed_tubule_metadata.csv')
            processed_df.to_csv(metadata_path, index=False)
            print(f"\n✓ Saved processed metadata: {metadata_path}")
            print(f"✓ Total patches extracted: {len(processed_df)}")
            print(f"\nState distribution:\n{processed_df['state'].value_counts().to_string()}")
            print(f"\nDataset distribution:\n{processed_df['dataset'].value_counts().to_string()}")
        
        print("\n" + "="*70)
        print("✓ EXTRACTION COMPLETE!")
        print("="*70 + "\n")




if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-CHANNEL CODEX PATCH EXTRACTION (ROBUST)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Metadata: {CONFIG['metadata_file']}")
    print(f"  Patch size: {CONFIG['patch_size']}")
    print(f"  Output: {CONFIG['output_dir']}")
    print(f"  Markers: {CONFIG['markers']}")
    if CONFIG['img_list']:
        print(f"  Images: {len(CONFIG['img_list'])} specified")
    else:
        print(f"  Images: All in metadata")
    print("="*70)
    
    process_balanced_datasets(CONFIG)