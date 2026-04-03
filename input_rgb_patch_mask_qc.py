#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 14:35:23 2025

@author: thaocao
QUALITY CONTROL - VISUALIZE H&E AND CODEX PATCH PAIRS
Plot random pairs of extracted patches to verify extraction quality
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path

CONFIG = {
    # Directory containing extracted patches 
    # NOTE: might be hidden files
    'data_dir': './data/lupus_nephritis',
    
    # Patch size (should match what you extracted)
    'patch_size': 128,
    
    # Number of random pairs to display
    'n_samples': 2,
    
    # Layout: (rows, cols) for subplot grid
    # For 6 samples: (3, 2) means 3 rows, 2 columns
    # Each row shows one tubule pair (H&E on left, CODEX on right)
    'layout': (3, 2),
    
    # Figure size
    'figsize': (12, 15),
    
    # Optional: Specify exact files to view (set to None for random)
    'specific_files': None,
}


def plot_patch_pairs(data_dir, patch_size=128, n_samples=6, layout=(3, 2), 
                     figsize=(12, 15), specific_files=None):
    """
    Plot random pairs of H&E and CODEX patches for quality control.
    
    Args:
        data_dir: Directory containing he_patches_X/ and codex_patches_X/ folders
        patch_size: Size of patches (for folder names)
        n_samples: Number of random pairs to display
        layout: (rows, cols) for subplot grid
        figsize: Figure size
        specific_files: List of specific filenames to view, or None for random
    """
    
    # Construct paths
    he_dir = os.path.join(data_dir, f'he_patches_{patch_size}')
    codex_dir = os.path.join(data_dir, f'codex_patches_{patch_size}')
    
    # Check if directories exist
    if not os.path.exists(he_dir):
        print(f"❌ ERROR: H&E directory not found: {he_dir}")
        return
    
    if not os.path.exists(codex_dir):
        print(f"❌ ERROR: CODEX directory not found: {codex_dir}")
        return
    
    # Get list of files
    he_files = sorted([f for f in os.listdir(he_dir) if f.endswith('.npy')])
    
    if len(he_files) == 0:
        print(f"❌ ERROR: No .npy files found in {he_dir}")
        return
    
    print(f"✓ Found {len(he_files)} H&E patches")
    
    # Select files to display
    if specific_files is not None:
        # Use specific files
        selected_files = [f for f in specific_files if f in he_files]
        if len(selected_files) < len(specific_files):
            print(f"⚠️  Warning: Some specified files not found")
        n_samples = len(selected_files)
    else:
        # Random selection
        if n_samples > len(he_files):
            n_samples = len(he_files)
            print(f"⚠️  Only {len(he_files)} patches available, showing all")
        selected_files = random.sample(he_files, n_samples)
    
    print(f"✓ Displaying {n_samples} random patch pairs\n")
    
    # Create figure
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes for easier iteration
    if n_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each pair
    for idx, filename in enumerate(selected_files):
        if idx >= len(axes):
            break
        
        # Load H&E patch
        he_path = os.path.join(he_dir, filename)
        he_patch = np.load(he_path)
        
        # Load CODEX patch
        codex_path = os.path.join(codex_dir, filename)
        codex_patch = np.load(codex_path)
        
        # Parse filename for metadata
        # Format: LuN_A3_Area4_TID1523_CNormal_SHealthy_X2450_Y3891.npy
        parts = filename.replace('.npy', '').split('_')
        
        # Extract key info
        sample_info = filename.replace('.npy', '')
        
        # Try to extract tubule ID, classification, state
        try:
            tid = [p for p in parts if p.startswith('TID')][0]
            classification = [p for p in parts if p.startswith('C') and not p.startswith('Centroid')][0][1:]
            state = [p for p in parts if p.startswith('S')][0][1:]
            short_label = f"{tid}, {classification}, {state}"
        except:
            short_label = sample_info[:40] + "..."
        
        # Option: showing side by side layout versus stacked on single column
        if cols == 2:
            # Side by side layout
            ax_idx = idx * 2
            
            # H&E on left
            axes[ax_idx].imshow(he_patch)
            axes[ax_idx].set_title(f'H&E\n{short_label}', fontsize=9)
            axes[ax_idx].axis('off')
            
            # CODEX on right
            axes[ax_idx + 1].imshow(codex_patch)
            axes[ax_idx + 1].set_title(f'CODEX\n{short_label}', fontsize=9)
            axes[ax_idx + 1].axis('off')
        else:
            # Stacked or single column layout
            axes[idx].imshow(he_patch)
            axes[idx].set_title(f'{short_label}\nH&E', fontsize=9)
            axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(len(selected_files) * 2 if cols == 2 else len(selected_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Quality Control: {n_samples} Random Tubule Patches', 
                 fontsize=14, y=0.995)
    plt.show()
    
    print("="*70)
    print("Displayed patches:")
    for i, f in enumerate(selected_files, 1):
        print(f"  {i}. {f}")
    print("="*70)


def plot_with_mask(data_dir, patch_size=128, n_samples=3, figsize=(15, 5)):
    """
    Plot H&E, CODEX, and mask for quality control (3 columns).
    
    Args:
        data_dir: Directory containing patch folders
        patch_size: Size of patches
        n_samples: Number of samples to display (rows)
        figsize: Figure size
    """
    
    # Construct paths
    he_dir = os.path.join(data_dir, f'he_patches_{patch_size}')
    codex_dir = os.path.join(data_dir, f'codex_patches_{patch_size}')
    mask_dir = os.path.join(data_dir, f'tubule_masks_{patch_size}')
    
    # Check directories
    for d, name in [(he_dir, 'H&E'), (codex_dir, 'CODEX'), (mask_dir, 'Mask')]:
        if not os.path.exists(d):
            print(f"❌ ERROR: {name} directory not found: {d}")
            return
    
    # Get files
    he_files = sorted([f for f in os.listdir(he_dir) if f.endswith('.npy')])
    
    if len(he_files) == 0:
        print(f"❌ ERROR: No patches found")
        return
    
    # Random selection
    if n_samples > len(he_files):
        n_samples = len(he_files)
    selected_files = random.sample(he_files, n_samples)
    
    print(f"✓ Displaying {n_samples} tubules with H&E, CODEX, and masks\n")
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 3, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each sample
    for row, filename in enumerate(selected_files):
        # Load all three
        he_patch = np.load(os.path.join(he_dir, filename))
        codex_patch = np.load(os.path.join(codex_dir, filename))
        mask_patch = np.load(os.path.join(mask_dir, filename))
        
        # Parse filename
        parts = filename.replace('.npy', '').split('_')
        try:
            tid = [p for p in parts if p.startswith('TID')][0]
            classification = [p for p in parts if p.startswith('C') and not p.startswith('Centroid')][0][1:]
            state = [p for p in parts if p.startswith('S')][0][1:]
            label = f"{tid}\n{classification}, {state}"
        except:
            label = filename[:30]
        
        # H&E
        axes[row, 0].imshow(he_patch)
        if row == 0:
            axes[row, 0].set_title(f'H&E\n{label}', fontsize=10)
        else:
            axes[row, 0].set_title(label, fontsize=10)
        axes[row, 0].axis('off')
        
        # CODEX
        axes[row, 1].imshow(codex_patch)
        if row == 0:
            axes[row, 1].set_title('CODEX', fontsize=10)
        axes[row, 1].axis('off')
        
        # Mask
        axes[row, 2].imshow(mask_patch, cmap='gray')
        if row == 0:
            axes[row, 2].set_title('Mask', fontsize=10)
        axes[row, 2].axis('off')
        
        # Add mask statistics
        mask_area = np.sum(mask_patch)
        axes[row, 2].text(0.5, -0.15, f'Pixels: {mask_area}', 
                          ha='center', transform=axes[row, 2].transAxes, 
                          fontsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Quality Control: H&E + CODEX + Mask', fontsize=14, y=0.998)
    plt.show()
    
    print("="*70)
    print("Displayed patches:")
    for i, f in enumerate(selected_files, 1):
        print(f"  {i}. {f}")
    print("="*70)


def quick_stats(data_dir, patch_size=128):
    """
    Print quick statistics about extracted patches.
    
    Args:
        data_dir: Directory containing patch folders
        patch_size: Size of patches
    """
    
    he_dir = os.path.join(data_dir, f'he_patches_{patch_size}')
    codex_dir = os.path.join(data_dir, f'codex_patches_{patch_size}')
    mask_dir = os.path.join(data_dir, f'tubule_masks_{patch_size}')
    
    print("\n" + "="*70)
    print("PATCH EXTRACTION STATISTICS")
    print("="*70)
    
    for folder, name in [(he_dir, 'H&E'), (codex_dir, 'CODEX'), (mask_dir, 'Masks')]:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.npy')]
            print(f"{name:12s}: {len(files):5d} patches")
            
            if len(files) > 0:
                # Check a sample file
                sample = np.load(os.path.join(folder, files[0]))
                print(f"             Shape: {sample.shape}, dtype: {sample.dtype}")
        else:
            print(f"{name:12s}: Directory not found")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Print quick statistics
    quick_stats(CONFIG['data_dir'], CONFIG['patch_size'])
    
    # Plot H&E and CODEX pairs
    print("\nGenerating H&E + CODEX visualization...\n")
    plot_patch_pairs(
        data_dir=CONFIG['data_dir'],
        patch_size=CONFIG['patch_size'],
        n_samples=CONFIG['n_samples'],
        layout=CONFIG['layout'],
        figsize=CONFIG['figsize'],
        specific_files=CONFIG['specific_files']
    )
    
  
    print("\nGenerating H&E + CODEX + Mask visualization...\n")
    plot_with_mask(
        data_dir=CONFIG['data_dir'],
        patch_size=CONFIG['patch_size'],
        n_samples=3,
        figsize=(15, 5)
    )
    
    print("\n✓ Visualization complete!")