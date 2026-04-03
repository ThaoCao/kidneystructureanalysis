#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:20:36 2025

@author: thaocao

CONTRASTIVE LEARNING FOR TUBULE CLASSIFICATION

Trains a SimCLR-style contrastive learning model and generates plots of:
- Training and validation loss
- Training and validation accuracy (for fine-tuning)
- F1 scores (for fine-tuning)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    gpu_id = 0
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
else:
    gpu_id = None
    print("⚠️  No GPU found, will use CPU")

# Auto-detect optimal number of workers
num_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 4
print(f"✓ Using {num_workers} workers")


CONFIG = {
    # ===== Data Paths =====
    'data_dir': './data',          # Root directory with patches
    'metadata_file': './data/processed_tubule_metadata.csv',
    'patch_size': 64,                             # Must match extracted patches
    
    # ===== Input Channels =====
    'he_channels': 3,                              # H&E RGB channels
    'codex_channels': 4,                           # CODEX marker channels (CD10, Claudin1, MUC1, CD45)
    'codex_channel_names': ['CD10', 'Claudin1', 'MUC1', 'CD45'],  # For reference
    
    # ===== Training Split =====
    'train_fraction': 0.8,                         # 80% train, 20% val
    'random_seed': 42,                             # For reproducibility
    
    # ===== Model Architecture =====
    'num_layers': 4,                               # CNN encoder layers
    'num_filters': 64,                             # Initial filters
    'embedding_size': 128,                         # Embedding dimension
    'projection_dim': 64,                          # Projection head dimension
    'batch_norm': True,                            # Use batch normalization
    
    # ===== Training Hyperparameters =====
    'phase': 'finetune',                        # 'contrastive' or 'finetune'
    'batch_size': 16,                              # Batch size
    'num_epochs': 100,                             # Training epochs
    'lr': 1e-3,                                    # Learning rate
    'temperature': 0.5,                            # Contrastive loss temperature
    
    # ===== Data Augmentation =====
    'use_augmentation': True,                      # Random flips/rotations
    'normalize_01': True,                          # Normalize to [0, 1]
    'validate_files': True,                        # Validate file loading and shapes (but keep empty channels)
    'save_invalid_list': True,                     # Save list of invalid (unloadable) patches
    
    # ===== Fine-tuning (only for phase='finetune') =====
    'contrastive_checkpoint': '/home/thaocao/Palom/logs/checkpoints/contrastive-epoch=59-val_loss=2.3517.ckpt',                # Path to pretrained model
    'freeze_encoder': False,                       # Freeze encoder weights
    'dropout': 0.3,                                # Dropout for classifier
    
    # ===== Hardware =====
    'num_workers': num_workers,                    # DataLoader workers (auto-detected)
    'gpu_id': gpu_id,                              # Which GPU to use (auto-detected)
    
    # ===== Logging =====
    'log_dir': './logs',                           # TensorBoard logs
    'save_checkpoints': True,                      # Save model checkpoints
    'checkpoint_every_n_epochs': 10,               # Save frequency
    
    # ===== Plotting =====
    'plot_results': True,                          # Generate plots at end
    'save_plots': True,                            # Save plots to disk
}


# METRICS TRACKING CALLBACK
class MetricsCallback(Callback):
    """Callback to track metrics for plotting"""
    
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_f1s = []
        self.epochs = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Get logged metrics
        metrics = trainer.callback_metrics
        
        # Track epoch
        epoch = trainer.current_epoch
        self.epochs.append(epoch)
        
        # Track losses
        if 'train_loss_epoch' in metrics:
            self.train_losses.append(metrics['train_loss_epoch'].item())
        
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'].item())
        
        # Track accuracies (for fine-tuning)
        if 'train_acc_epoch' in metrics:
            self.train_accs.append(metrics['train_acc_epoch'].item())
        
        if 'val_acc' in metrics:
            self.val_accs.append(metrics['val_acc'].item())
        
        if 'val_f1' in metrics:
            self.val_f1s.append(metrics['val_f1'].item())


# PLOTTING FUNCTIONS

def plot_training_curves(metrics_callback, phase='contrastive', save_path=None):
    """
    Plot training curves for loss and accuracy
    
    Args:
        metrics_callback: MetricsCallback with tracked metrics
        phase: 'contrastive' or 'finetune'
        save_path: Path to save plot (optional)
    """
    epochs = metrics_callback.epochs
    
    if phase == 'contrastive':
        # Plot only losses for contrastive learning
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(epochs, metrics_callback.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, metrics_callback.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Contrastive Learning - Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add best val loss annotation
        best_val_loss = min(metrics_callback.val_losses)
        best_epoch = metrics_callback.val_losses.index(best_val_loss)
        ax.axvline(x=epochs[best_epoch], color='g', linestyle='--', alpha=0.5, label=f'Best Val (epoch {epochs[best_epoch]})')
        ax.legend(fontsize=11)
        
    else:  # finetune
        # Plot losses and accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss plot
        ax1.plot(epochs, metrics_callback.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, metrics_callback.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, metrics_callback.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, metrics_callback.val_accs, 'r-', label='Val Accuracy', linewidth=2)
        if len(metrics_callback.val_f1s) > 0:
            ax2.plot(epochs, metrics_callback.val_f1s, 'g--', label='Val F1 Score', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Accuracy & F1 Score', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Add best accuracy annotation
        best_val_acc = max(metrics_callback.val_accs)
        best_epoch = metrics_callback.val_accs.index(best_val_acc)
        ax2.axvline(x=epochs[best_epoch], color='purple', linestyle='--', alpha=0.5)
        ax2.text(epochs[best_epoch], 0.95, f'Best: {best_val_acc:.3f}\n(epoch {epochs[best_epoch]})',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Fine-tuning - Training Progress', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Initial train loss: {metrics_callback.train_losses[0]:.4f}")
    print(f"Final train loss:   {metrics_callback.train_losses[-1]:.4f}")
    print(f"Best val loss:      {min(metrics_callback.val_losses):.4f} (epoch {metrics_callback.val_losses.index(min(metrics_callback.val_losses))})")
    
    if phase == 'finetune':
        print(f"\nInitial train acc:  {metrics_callback.train_accs[0]:.4f}")
        print(f"Final train acc:    {metrics_callback.train_accs[-1]:.4f}")
        print(f"Best val acc:       {max(metrics_callback.val_accs):.4f} (epoch {metrics_callback.val_accs.index(max(metrics_callback.val_accs))})")
        if len(metrics_callback.val_f1s) > 0:
            print(f"Best val F1:        {max(metrics_callback.val_f1s):.4f} (epoch {metrics_callback.val_f1s.index(max(metrics_callback.val_f1s))})")
    print("="*70)

# DATA LOADING

class TubuleDataset(Dataset):
    """Dataset for loading tubule patches from .npy files"""
    
    def __init__(self, metadata_df, data_dir, patch_size=64, 
                 normalize_01=True, augment=False, validate_files=True,
                 save_invalid_list=False, invalid_list_path=None):
        """
        Args:
            metadata_df: DataFrame with columns: sample_area, tubule_id, 
                        classification, centroid_x, centroid_y, filename
            data_dir: Root directory containing he_patches_X/ and codex_patches_X/
            patch_size: Size of patches
            normalize_01: Normalize images to [0, 1]
            augment: Apply random augmentations
            validate_files: Validate that files can be loaded and have correct shapes
            save_invalid_list: Save list of invalid (unloadable) patches to file
            invalid_list_path: Path to save invalid patches list
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.normalize_01 = normalize_01
        self.augment = augment
        self.validate_files = validate_files
        self.save_invalid_list = save_invalid_list
        self.invalid_list_path = invalid_list_path
        
        # Paths
        self.he_dir = os.path.join(data_dir, f'he_patches_{patch_size}')
        self.codex_dir = os.path.join(data_dir, f'codex_patches_{patch_size}')
        
        # Map classification labels to integers
        self.label_to_idx = self._create_label_mapping()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        print(f"  Dataset: {len(self.metadata)} samples (before filtering)")
        
        # Initialize channel counts (will be verified from sample)
        self.he_channels = None
        self.codex_channels = None
        
        # Check a sample to get channel info
        if len(self.metadata) > 0:
            sample_filename = self.metadata.iloc[0]['filename']
            if not sample_filename.endswith('.npy'):
                sample_filename = sample_filename + '.npy'
            
            he_sample = np.load(os.path.join(self.he_dir, sample_filename))
            codex_sample = np.load(os.path.join(self.codex_dir, sample_filename))
            print(f"  H&E shape: {he_sample.shape}, dtype: {he_sample.dtype}")
            print(f"  CODEX shape: {codex_sample.shape}, dtype: {codex_sample.dtype}")
            self.he_channels = he_sample.shape[2] if he_sample.ndim == 3 else 1
            self.codex_channels = codex_sample.shape[2] if codex_sample.ndim == 3 else 1
            print(f"  H&E channels: {self.he_channels}, CODEX channels: {self.codex_channels}")
        
        # Validate files (filter only truly invalid files, not empty channels)
        if self.validate_files:
            print(f"  Validating file integrity...")
            self.metadata, invalid_info = self._validate_files()
            print(f"  Dataset: {len(self.metadata)} samples (after validation)")
            
            # Save invalid list if requested
            if self.save_invalid_list and len(invalid_info) > 0 and self.invalid_list_path:
                self._save_invalid_list(invalid_info)
        else:
            print(f"  Skipping file validation (disabled in config)")
        
    def _create_label_mapping(self):
        """Create mapping from tubule classification to integer"""
        unique_labels = sorted(self.metadata['state'].unique())
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def _validate_files(self):
        """
        Validate that files can be loaded and have correct shapes.
        Empty channels (all zeros) are kept as they represent valid biological data.
        
        Returns:
            Tuple of (validated_metadata, invalid_patches_info)
        """
        valid_indices = []
        invalid_patches = []
        empty_channel_count = 0
        
        print(f"    Validating {len(self.metadata)} patches...")
        
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            filename = row['filename']
            if not filename.endswith('.npy'):
                filename = filename + '.npy'
            
            is_valid = True
            error_msg = None
            
            try:
                # Try to load H&E patch
                he_path = os.path.join(self.he_dir, filename)
                if not os.path.exists(he_path):
                    is_valid = False
                    error_msg = "H&E file not found"
                else:
                    he_patch = np.load(he_path)
                    
                    # Check H&E shape
                    if he_patch.ndim != 3:
                        is_valid = False
                        error_msg = f"H&E wrong dimensions: {he_patch.shape}"
                    elif he_patch.shape[0] != self.patch_size or he_patch.shape[1] != self.patch_size:
                        is_valid = False
                        error_msg = f"H&E wrong size: {he_patch.shape}"
                
                # Try to load CODEX patch
                if is_valid:
                    codex_path = os.path.join(self.codex_dir, filename)
                    if not os.path.exists(codex_path):
                        is_valid = False
                        error_msg = "CODEX file not found"
                    else:
                        codex_patch = np.load(codex_path)
                        
                        # Check CODEX shape
                        if codex_patch.ndim != 3:
                            is_valid = False
                            error_msg = f"CODEX wrong dimensions: {codex_patch.shape}"
                        elif codex_patch.shape[0] != self.patch_size or codex_patch.shape[1] != self.patch_size:
                            is_valid = False
                            error_msg = f"CODEX wrong size: {codex_patch.shape}"
                        elif codex_patch.shape[2] != self.codex_channels:
                            is_valid = False
                            error_msg = f"CODEX wrong channel count: {codex_patch.shape[2]}, expected {self.codex_channels}"
                        else:
                            # Check for empty channels (but don't filter them out)
                            for ch in range(codex_patch.shape[2]):
                                ch_data = codex_patch[:, :, ch]
                                if np.max(ch_data) == 0 and np.min(ch_data) == 0:
                                    empty_channel_count += 1
                
                if is_valid:
                    valid_indices.append(idx)
                else:
                    invalid_patches.append({
                        'filename': row['filename'],
                        'tubule_id': row.get('tubule_id', 'N/A'),
                        'sample_area': row.get('sample_area', 'N/A'),
                        'state': row.get('state', 'N/A'),
                        'error': error_msg
                    })
                    
            except Exception as e:
                # If we can't load the file, filter it out
                invalid_patches.append({
                    'filename': row['filename'],
                    'tubule_id': row.get('tubule_id', 'N/A'),
                    'sample_area': row.get('sample_area', 'N/A'),
                    'state': row.get('state', 'N/A'),
                    'error': str(e)
                })
        
        # Create validated dataframe
        validated_metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        
        # Report validation results
        n_invalid = len(self.metadata) - len(validated_metadata)
        if n_invalid > 0:
            print(f"    ⚠️  Filtered out {n_invalid} invalid patches:")
            for i, info in enumerate(invalid_patches[:10]):  # Show first 10
                print(f"      {i+1}. {info['filename']} - {info['error']}")
            if len(invalid_patches) > 10:
                print(f"      ... and {len(invalid_patches) - 10} more")
        else:
            print(f"    ✓ All patches are valid")
        
        if empty_channel_count > 0:
            print(f"    ℹ️  Note: Found {empty_channel_count} empty channels (all zeros) across patches")
            print(f"       These are kept as they represent valid biological data (marker absence)")
        
        return validated_metadata, invalid_patches
    
    def _save_invalid_list(self, invalid_info):
        """Save list of invalid patches to CSV file."""
        try:
            df = pd.DataFrame(invalid_info)
            df.to_csv(self.invalid_list_path, index=False)
            print(f"    ✓ Saved invalid patches list to: {self.invalid_list_path}")
        except Exception as e:
            print(f"    ⚠️  Could not save invalid list: {e}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        try:
            row = self.metadata.iloc[idx]
            
            # Handle filename - check if .npy extension already exists
            filename = row['filename']
            if not filename.endswith('.npy'):
                filename = filename + '.npy'
            
            # Load H&E patch
            he_path = os.path.join(self.he_dir, filename)
            if not os.path.exists(he_path):
                raise FileNotFoundError(f"H&E file not found: {he_path}")
            
            he_patch = np.load(he_path).astype(np.float32)
            
            # Load CODEX patch
            codex_path = os.path.join(self.codex_dir, filename)
            if not os.path.exists(codex_path):
                raise FileNotFoundError(f"CODEX file not found: {codex_path}")
            
            codex_patch = np.load(codex_path).astype(np.float32)
            
            # Validate shapes
            if he_patch.ndim != 3:
                raise ValueError(f"H&E patch has wrong dimensions: {he_patch.shape}, expected (H, W, C)")
            if codex_patch.ndim != 3:
                raise ValueError(f"CODEX patch has wrong dimensions: {codex_patch.shape}, expected (H, W, C)")
            
            # Normalize
            if self.normalize_01:
                # Normalize H&E (uint8 RGB, 0-255)
                if he_patch.max() > 1.0:  # If not already normalized
                    he_patch = he_patch / 255.0
                
                # Normalize CODEX (may be uint16 or float, different range per channel)
                # Use percentile-based normalization per channel
                if codex_patch.ndim == 3:
                    for ch in range(codex_patch.shape[2]):
                        ch_data = codex_patch[:, :, ch]
                        ch_min = float(ch_data.min())
                        ch_max = float(np.percentile(ch_data, 99.9))  # Use 99.9th percentile to avoid outliers
                        
                        if ch_max > ch_min:
                            codex_patch[:, :, ch] = (ch_data - ch_min) / (ch_max - ch_min)
                        else:
                            # All values are the same, just set to 0
                            codex_patch[:, :, ch] = 0.0
                else:
                    # Single channel case
                    ch_min = float(codex_patch.min())
                    ch_max = float(np.percentile(codex_patch, 99.9))
                    if ch_max > ch_min:
                        codex_patch = (codex_patch - ch_min) / (ch_max - ch_min)
                    else:
                        codex_patch = codex_patch * 0.0
            
            # Convert to torch tensors (H, W, C) -> (C, H, W)
            he_patch = torch.from_numpy(he_patch).permute(2, 0, 1)
            codex_patch = torch.from_numpy(codex_patch).permute(2, 0, 1)
            
            # Validate tensor shapes (if channel counts are known)
            if self.he_channels is not None and he_patch.shape[0] != self.he_channels:
                raise ValueError(f"H&E has {he_patch.shape[0]} channels, expected {self.he_channels}")
            if self.codex_channels is not None and codex_patch.shape[0] != self.codex_channels:
                raise ValueError(f"CODEX has {codex_patch.shape[0]} channels, expected {self.codex_channels}")
            
            # Apply same augmentation to both
            if self.augment:
                he_patch, codex_patch = self._augment_pair(he_patch, codex_patch)
            
            # Get label
            label = self.label_to_idx[row['state']]
            
            return he_patch, codex_patch, label, row['tubule_id']
            
        except Exception as e:
            print(f"\n❌ ERROR loading sample {idx}:")
            print(f"   Filename: {row['filename']}")
            print(f"   Tubule ID: {row.get('tubule_id', 'N/A')}")
            print(f"   Error: {str(e)}")
            raise
    
    def _augment_pair(self, he_patch, codex_patch):
        """Apply random flips and rotations to both patches"""
        # Random vertical flip
        if torch.rand(1) > 0.5:
            he_patch = TF.vflip(he_patch)
            codex_patch = TF.vflip(codex_patch)
        
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            he_patch = TF.hflip(he_patch)
            codex_patch = TF.hflip(codex_patch)
        
        # Random 90-degree rotation
        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle > 0:
            he_patch = TF.rotate(he_patch, angle)
            codex_patch = TF.rotate(codex_patch, angle)
        
        return he_patch, codex_patch


def create_train_val_split(metadata_file, train_fraction=0.8, random_seed=42):
    """Split metadata into train and validation sets"""
    metadata = pd.read_csv(metadata_file)
    
    # Shuffle
    metadata = metadata.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    split_idx = int(len(metadata) * train_fraction)
    train_df = metadata.iloc[:split_idx]
    val_df = metadata.iloc[split_idx:]
    
    print(f"Total tubules: {len(metadata)}")
    print(f"Train: {len(train_df)} ({len(train_df)/len(metadata)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(metadata)*100:.1f}%)")
    
    # Print class distribution
    print("\nTrain classification distribution:")
    print(train_df['state'].value_counts())
    print("\nVal classification distribution:")
    print(val_df['state'].value_counts())
    
    return train_df, val_df


def validate_dataset(metadata_file, data_dir, patch_size=64, num_samples=5):
    """
    Validate dataset by checking a few samples before training.
    
    Args:
        metadata_file: Path to metadata CSV
        data_dir: Root data directory
        patch_size: Patch size
        num_samples: Number of samples to validate
    """
    print("\n" + "="*70)
    print("DATASET VALIDATION")
    print("="*70)
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    print(f"✓ Loaded metadata: {len(metadata)} rows")
    print(f"  Columns: {metadata.columns.tolist()}")
    
    # Check directories
    he_dir = os.path.join(data_dir, f'he_patches_{patch_size}')
    codex_dir = os.path.join(data_dir, f'codex_patches_{patch_size}')
    
    if not os.path.exists(he_dir):
        print(f"❌ H&E directory not found: {he_dir}")
        return False
    if not os.path.exists(codex_dir):
        print(f"❌ CODEX directory not found: {codex_dir}")
        return False
    
    print(f"✓ H&E directory exists: {he_dir}")
    print(f"✓ CODEX directory exists: {codex_dir}")
    
    # Check files
    he_files = [f for f in os.listdir(he_dir) if f.endswith('.npy')]
    codex_files = [f for f in os.listdir(codex_dir) if f.endswith('.npy')]
    print(f"✓ Found {len(he_files)} H&E files")
    print(f"✓ Found {len(codex_files)} CODEX files")
    
    # Validate samples
    print(f"\nValidating {num_samples} random samples...")
    sample_indices = np.random.choice(len(metadata), min(num_samples, len(metadata)), replace=False)
    
    issues = []
    for i, idx in enumerate(sample_indices):
        row = metadata.iloc[idx]
        filename = row['filename']
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        
        print(f"\n  Sample {i+1}/{len(sample_indices)}: {filename}")
        
        # Check H&E
        he_path = os.path.join(he_dir, filename)
        if not os.path.exists(he_path):
            issue = f"H&E file missing: {filename}"
            print(f"    ❌ {issue}")
            issues.append(issue)
            continue
        
        try:
            he_patch = np.load(he_path)
            print(f"    ✓ H&E loaded: shape={he_patch.shape}, dtype={he_patch.dtype}, "
                  f"min={he_patch.min():.2f}, max={he_patch.max():.2f}")
        except Exception as e:
            issue = f"H&E load error for {filename}: {e}"
            print(f"    ❌ {issue}")
            issues.append(issue)
            continue
        
        # Check CODEX
        codex_path = os.path.join(codex_dir, filename)
        if not os.path.exists(codex_path):
            issue = f"CODEX file missing: {filename}"
            print(f"    ❌ {issue}")
            issues.append(issue)
            continue
        
        try:
            codex_patch = np.load(codex_path)
            print(f"    ✓ CODEX loaded: shape={codex_patch.shape}, dtype={codex_patch.dtype}, "
                  f"min={codex_patch.min():.2f}, max={codex_patch.max():.2f}")
            
            # Check each CODEX channel
            if codex_patch.ndim == 3:
                for ch in range(codex_patch.shape[2]):
                    ch_data = codex_patch[:, :, ch]
                    print(f"      Ch{ch}: min={ch_data.min():.2f}, max={ch_data.max():.2f}, "
                          f"mean={ch_data.mean():.2f}, non-zero={np.count_nonzero(ch_data)}")
        except Exception as e:
            issue = f"CODEX load error for {filename}: {e}"
            print(f"    ❌ {issue}")
            issues.append(issue)
            continue
        
        # Check shapes
        if he_patch.ndim != 3:
            issue = f"H&E wrong dimensions: {he_patch.shape}"
            print(f"    ⚠️  {issue}")
            issues.append(issue)
        
        if codex_patch.ndim != 3:
            issue = f"CODEX wrong dimensions: {codex_patch.shape}"
            print(f"    ⚠️  {issue}")
            issues.append(issue)
    
    # Summary
    print("\n" + "="*70)
    if len(issues) == 0:
        print("✓ VALIDATION PASSED - All samples loaded successfully!")
        return True
    else:
        print(f"⚠️  VALIDATION FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False

# MODEL ARCHITECTURE

class CNNEncoder(nn.Module):
    """CNN encoder for tubule patches"""
    
    def __init__(self, input_channels=3, input_size=64, num_layers=4, 
                 num_filters=64, embedding_size=64, batch_norm=True):
        super(CNNEncoder, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, 
                                   kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU())
            if batch_norm:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters
            num_filters *= 2
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flat_size = self._get_flat_size(input_channels, input_size)
        
        # Fully connected layer
        self.fc = nn.Linear(self.flat_size, embedding_size)
    
    def _get_flat_size(self, channels, size):
        """Calculate size after CNN layers"""
        x = torch.rand(1, channels, size, size)
        x = self.cnn(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ContrastiveLoss(nn.Module):
    """SimCLR-style contrastive loss"""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_he, z_codex):
        """
        Args:
            z_he: H&E embeddings [batch_size, embedding_dim]
            z_codex: CODEX embeddings [batch_size, embedding_dim]
        """
        batch_size = z_he.shape[0]
        
        # Normalize
        z_he = F.normalize(z_he, p=2, dim=1)
        z_codex = F.normalize(z_codex, p=2, dim=1)
        
        # Concatenate
        representations = torch.cat([z_he, z_codex], dim=0)
        
        # Compute similarity matrix
        similarity = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )
        
        # Mask to remove self-similarity
        mask = (~torch.eye(batch_size * 2, dtype=bool, device=z_he.device)).float()
        
        # Positive pairs
        sim_he_codex = torch.diag(similarity, batch_size)
        sim_codex_he = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_he_codex, sim_codex_he], dim=0)
        
        # Compute loss
        nominator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)
        
        loss = -torch.log(nominator / torch.sum(denominator, dim=1))
        return loss.mean()


class SimCLRModel(nn.Module):
    """Complete SimCLR model with encoders and projection head for multi-channel inputs"""
    
    def __init__(self, he_channels=3, codex_channels=4, input_size=64, num_layers=4,
                 num_filters=64, embedding_size=64, projection_dim=64,
                 batch_norm=True):
        super(SimCLRModel, self).__init__()
        
        print(f"\nInitializing SimCLR Model:")
        print(f"  H&E channels: {he_channels}")
        print(f"  CODEX channels: {codex_channels}")
        print(f"  Embedding size: {embedding_size}")
        print(f"  Projection dim: {projection_dim}")
        
        # Separate initial conv for H&E and CODEX (different input channels)
        self.he_conv = nn.Sequential(
            nn.Conv2d(he_channels, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_filters) if batch_norm else nn.Identity()
        )
        
        self.codex_conv = nn.Sequential(
            nn.Conv2d(codex_channels, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_filters) if batch_norm else nn.Identity()
        )
        
        # Shared encoder (after initial conv, both have num_filters channels)
        self.encoder = CNNEncoder(
            input_channels=num_filters,
            input_size=input_size,
            num_layers=num_layers-1,  # -1 because we have initial conv
            num_filters=num_filters,
            embedding_size=embedding_size,
            batch_norm=batch_norm
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, projection_dim)
        )
    
    def forward(self, he_img, codex_img):
        # Process H&E (3 channels)
        he_feat = self.he_conv(he_img)
        he_embed = self.encoder(he_feat)
        he_proj = self.projection(he_embed)
        
        # Process CODEX (4 channels)
        codex_feat = self.codex_conv(codex_img)
        codex_embed = self.encoder(codex_feat)
        codex_proj = self.projection(codex_embed)
        
        return he_embed, codex_embed, he_proj, codex_proj


# PYTORCH LIGHTNING MODULES


class ContrastiveLearningModule(pl.LightningModule):
    """Lightning module for contrastive learning with multi-channel inputs"""
    
    def __init__(self, he_channels=3, codex_channels=4, input_size=64, num_layers=4,
                 num_filters=64, embedding_size=64, projection_dim=64,
                 batch_norm=True, temperature=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SimCLRModel(
            he_channels=he_channels,
            codex_channels=codex_channels,
            input_size=input_size,
            num_layers=num_layers,
            num_filters=num_filters,
            embedding_size=embedding_size,
            projection_dim=projection_dim,
            batch_norm=batch_norm
        )
        
        self.loss_fn = ContrastiveLoss(temperature=temperature)
        self.lr = lr
    
    def forward(self, he_img, codex_img):
        return self.model(he_img, codex_img)
    
    def training_step(self, batch, batch_idx):
        he_img, codex_img, labels, tubule_ids = batch
        
        he_embed, codex_embed, he_proj, codex_proj = self.model(he_img, codex_img)
        
        loss = self.loss_fn(he_proj, codex_proj)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, 
                prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        he_img, codex_img, labels, tubule_ids = batch
        
        he_embed, codex_embed, he_proj, codex_proj = self.model(he_img, codex_img)
        
        loss = self.loss_fn(he_proj, codex_proj)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class SupervisedModule(pl.LightningModule):
    """Lightning module for supervised fine-tuning"""
    
    def __init__(self, pretrained_model, num_classes, lr=1e-3, 
                 freeze_encoder=False, dropout=0.3):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        
        # Get pretrained encoder
        self.model = pretrained_model.model
        self.num_classes = num_classes
        self.lr = lr
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Classification head
        embedding_size = pretrained_model.hparams.embedding_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(embedding_size, num_classes)
        )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', 
                                                num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass',
                                              num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task='multiclass',
                                            num_classes=num_classes,
                                            average='weighted')
    
    def forward(self, he_img):
        # Get H&E embedding
        he_feat = self.model.he_conv(he_img)
        he_embed = self.model.encoder(he_feat)
        
        # Classify
        logits = self.classifier(he_embed)
        return logits
    
    def training_step(self, batch, batch_idx):
        he_img, codex_img, labels, tubule_ids = batch
        
        logits = self(he_img)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True,
                prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        he_img, codex_img, labels, tubule_ids = batch
        
        logits = self(he_img)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        f1 = self.val_f1(preds, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# TRAINING FUNCTIONS


def train_contrastive(config):
    """Train contrastive learning model"""
    
    print("\n" + "="*70)
    print("CONTRASTIVE LEARNING PHASE")
    print("="*70)
    
    # Create log directory
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_df, val_df = create_train_val_split(
        config['metadata_file'],
        config['train_fraction'],
        config['random_seed']
    )
    
    train_dataset = TubuleDataset(
        train_df,
        config['data_dir'],
        config['patch_size'],
        config['normalize_01'],
        config['use_augmentation'],
        validate_files=config['validate_files'],
        save_invalid_list=config['save_invalid_list'],
        invalid_list_path=os.path.join(config['log_dir'], 'invalid_patches_train.csv')
    )
    
    val_dataset = TubuleDataset(
        val_df,
        config['data_dir'],
        config['patch_size'],
        config['normalize_01'],
        augment=False,  # No augmentation for validation
        validate_files=config['validate_files'],
        save_invalid_list=config['save_invalid_list'],
        invalid_list_path=os.path.join(config['log_dir'], 'invalid_patches_val.csv')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = ContrastiveLearningModule(
        he_channels=config['he_channels'],
        codex_channels=config['codex_channels'],
        input_size=config['patch_size'],
        num_layers=config['num_layers'],
        num_filters=config['num_filters'],
        embedding_size=config['embedding_size'],
        projection_dim=config['projection_dim'],
        batch_norm=config['batch_norm'],
        temperature=config['temperature'],
        lr=config['lr']
    )
    
    # Setup logging
    logger = TensorBoardLogger(config['log_dir'], name='contrastive')
    
    # Setup callbacks
    callbacks = []
    
    # Metrics tracking callback
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)
    
    if config['save_checkpoints']:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join(config['log_dir'], 'checkpoints'),
            filename='contrastive-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min',
            every_n_epochs=config['checkpoint_every_n_epochs']
        )
        callbacks.append(checkpoint_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu' if config['gpu_id'] is not None else 'cpu',
        devices=[config['gpu_id']] if config['gpu_id'] is not None else 1,
        max_epochs=config['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✓ Contrastive learning complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path if config['save_checkpoints'] else 'N/A'}")
    
    # Plot results
    if config['plot_results']:
        save_path = os.path.join(config['log_dir'], 'contrastive_training_curves.png') if config['save_plots'] else None
        plot_training_curves(metrics_callback, phase='contrastive', save_path=save_path)
    
    return model, checkpoint_callback.best_model_path if config['save_checkpoints'] else None


def train_finetune(config):
    """Fine-tune for tubule classification"""
    
    print("\n" + "="*70)
    print("FINE-TUNING PHASE")
    print("="*70)
    
    # Create log directory
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_df, val_df = create_train_val_split(
        config['metadata_file'],
        config['train_fraction'],
        config['random_seed']
    )
    
    train_dataset = TubuleDataset(
        train_df,
        config['data_dir'],
        config['patch_size'],
        config['normalize_01'],
        config['use_augmentation'],
        validate_files=config['validate_files'],
        save_invalid_list=config['save_invalid_list'],
        invalid_list_path=os.path.join(config['log_dir'], 'invalid_patches_finetune_train.csv')
    )
    
    val_dataset = TubuleDataset(
        val_df,
        config['data_dir'],
        config['patch_size'],
        config['normalize_01'],
        augment=False,
        validate_files=config['validate_files'],
        save_invalid_list=config['save_invalid_list'],
        invalid_list_path=os.path.join(config['log_dir'], 'invalid_patches_finetune_val.csv')
    )
    
    num_classes = train_dataset.num_classes
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {train_dataset.label_to_idx}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Load pretrained model
    print(f"\nLoading pretrained model from: {config['contrastive_checkpoint']}")
    pretrained_model = ContrastiveLearningModule.load_from_checkpoint(
        config['contrastive_checkpoint']
    )
    
    # Create fine-tuning model
    print("\nCreating fine-tuning model...")
    model = SupervisedModule(
        pretrained_model,
        num_classes=num_classes,
        lr=config['lr'],
        freeze_encoder=config['freeze_encoder'],
        dropout=config['dropout']
    )
    
    # Setup logging
    logger = TensorBoardLogger(config['log_dir'], name='finetune')
    
    # Setup callbacks
    callbacks = []
    
    # Metrics tracking callback
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)
    
    if config['save_checkpoints']:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=os.path.join(config['log_dir'], 'checkpoints'),
            filename='finetune-{epoch:02d}-{val_acc:.4f}',
            save_top_k=3,
            mode='max',
            every_n_epochs=config['checkpoint_every_n_epochs']
        )
        callbacks.append(checkpoint_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu' if config['gpu_id'] is not None else 'cpu',
        devices=[config['gpu_id']] if config['gpu_id'] is not None else 1,
        max_epochs=config['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )
    
    # Train
    print("\nStarting fine-tuning...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✓ Fine-tuning complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path if config['save_checkpoints'] else 'N/A'}")
    
    # Plot results
    if config['plot_results']:
        save_path = os.path.join(config['log_dir'], 'finetune_training_curves.png') if config['save_plots'] else None
        plot_training_curves(metrics_callback, phase='finetune', save_path=save_path)
    
    return model, checkpoint_callback.best_model_path if config['save_checkpoints'] else None

# MAIN EXECUTION

if __name__ == "__main__":

    
    print("\n" + "="*70)
    print("🔬 TUBULE CLASSIFICATION - CONTRASTIVE LEARNING (MULTI-CHANNEL)")
    print("="*70)
    print(f"\nPhase: {CONFIG['phase'].upper()}")
    print(f"Data directory: {CONFIG['data_dir']}")
    print(f"Patch size: {CONFIG['patch_size']}")
    print(f"H&E channels: {CONFIG['he_channels']}")
    print(f"CODEX channels: {CONFIG['codex_channels']} {CONFIG['codex_channel_names']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"GPU: {'GPU ' + str(CONFIG['gpu_id']) if CONFIG['gpu_id'] is not None else 'CPU'}")
    print(f"Workers: {CONFIG['num_workers']}")
    print("="*70)
    
    # Verify paths
    if not os.path.exists(CONFIG['data_dir']):
        print(f"\n❌ ERROR: Data directory not found: {CONFIG['data_dir']}")
        print("Please run patch extraction first!")
    elif not os.path.exists(CONFIG['metadata_file']):
        print(f"\n❌ ERROR: Metadata file not found: {CONFIG['metadata_file']}")
        print("Please run patch extraction first!")
    else:
        # Validate dataset before training
        print("\nValidating dataset before training...")
        validation_passed = validate_dataset(
            CONFIG['metadata_file'],
            CONFIG['data_dir'],
            CONFIG['patch_size'],
            num_samples=5
        )
        
        if not validation_passed:
            print("\n❌ Dataset validation failed! Please fix the issues above before training.")
            print("You can check individual files using the validation output.")
            import sys
            sys.exit(1)
        
        # Run training
        if CONFIG['phase'] == 'contrastive':
            model, checkpoint_path = train_contrastive(CONFIG)
            
            print("\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print("1. Copy the checkpoint path above")
            print("2. Set CONFIG['phase'] = 'finetune'")
            print("3. Set CONFIG['contrastive_checkpoint'] = 'checkpoint_path'")
            print("4. Run again to fine-tune!")
            
        elif CONFIG['phase'] == 'finetune':
            if CONFIG['contrastive_checkpoint'] is None:
                print("\n❌ ERROR: Must set CONFIG['contrastive_checkpoint']")
                print("Please provide path to pretrained contrastive model")
            else:
                model, checkpoint_path = train_finetune(CONFIG)
                
                print("\n" + "="*70)
                print("TRAINING COMPLETE!")
                print("="*70)
                print("Your model is ready for tubule classification")
                print(f"Best checkpoint: {checkpoint_path}")
        else:
            print(f"\n❌ ERROR: Invalid phase '{CONFIG['phase']}'")
            print("Must be 'contrastive' or 'finetune'")