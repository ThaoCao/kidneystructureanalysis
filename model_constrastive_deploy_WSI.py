#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:56:27 2025

@author: thaocao

Deploy trained model on whole slide H&E images

Extracts patches for each tubule from whole slide image and predicts classification.
Updates dataframe with predictions matched by Sample_Area and TubuleID.

"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from tifffile import imread
import matplotlib.pyplot as plt


CONFIG = {
    # ===== Model Checkpoint =====
    'checkpoint_path': '/home/thaocao/Palom/logs/checkpoints/finetune-epoch=49-val_acc=0.9935.ckpt',
    
    # ===== Input Files =====
    'he_image_path': '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Lupus_Nephritis/DAPI_HE_normalized_aligned_rgb_cleaned_ds/012523S5_Area2.tif',
    'tubule_mask_path': '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Lupus_Nephritis/tubule_segmentations/masks/tubule_cleaned/cleaned/012523S5_Area2.tif',
    
    # ===== Metadata =====
    'metadata_file': '/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids.csv',
    'sample_name': 'LuN_012523S5_Area2',  
    
    # ===== Patch Parameters =====
    'patch_size': 64,              
    'batch_size': 32,               # For inference (larger = faster)
    'num_workers': 4,               # DataLoader workers
    
    # ===== Output =====
    'output_csv': './predictions/predicted_states.csv',
    'save_visualization': True,    
    'viz_output': './predictions/visualization_10232025_states.tif',
    
    # ===== Device =====
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

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


class SimCLRModel(nn.Module):
    """Complete SimCLR model with encoders and projection head"""
    
    def __init__(self, input_channels=3, input_size=64, num_layers=4,
                 num_filters=64, embedding_size=64, projection_dim=64,
                 batch_norm=True):
        super(SimCLRModel, self).__init__()
        
        # Separate initial conv for H&E and CODEX
        self.he_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        self.codex_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        # Shared encoder
        self.encoder = CNNEncoder(
            input_channels=num_filters,
            input_size=input_size,
            num_layers=num_layers,
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
        # Process H&E
        he_feat = self.he_conv(he_img)
        he_embed = self.encoder(he_feat)
        he_proj = self.projection(he_embed)
        
        # Process CODEX
        codex_feat = self.codex_conv(codex_img)
        codex_embed = self.encoder(codex_feat)
        codex_proj = self.projection(codex_embed)
        
        return he_embed, codex_embed, he_proj, codex_proj


# DATASET FOR WHOLE SLIDE INFERENCE

class WholeSlideInferenceDataset(Dataset):
    """Dataset for extracting patches from whole slide image"""
    
    def __init__(self, he_image, tubule_mask, patch_size=64):
        """
        Args:
            he_image: H&E image array (H, W, C) or (C, H, W)
            tubule_mask: Segmentation mask where each tubule has unique ID
            patch_size: Size of patches to extract
        """
        self.he_image = he_image
        self.tubule_mask = tubule_mask
        self.patch_size = patch_size
        
        
        # Ensure correct shape (H, W, C) for H&E
        if self.he_image.shape[0] == 3 or self.he_image.shape[0] == 4:
            self.he_image = np.transpose(self.he_image, (1, 2, 0))
        
        # Get list of tubule IDs
        self.tubule_ids = np.unique(tubule_mask)
        self.tubule_ids = self.tubule_ids[self.tubule_ids > 0]  # Exclude background
        self.tubule_ids = self.tubule_ids.astype(np.int32)
        # Calculate centroids for each tubule
        self.tubule_centroids = {}
        for tid in self.tubule_ids:
            coords = np.where(tubule_mask == tid)
            centroid_y = int(np.mean(coords[0]))
            centroid_x = int(np.mean(coords[1]))
            self.tubule_centroids[tid] = (centroid_x, centroid_y)
    
    def __len__(self):
        return len(self.tubule_ids)
    
    def __getitem__(self, idx):
        tubule_id = self.tubule_ids[idx]
        centroid_x, centroid_y = self.tubule_centroids[tubule_id]
        
        # Extract patch centered on tubule
        patch = self._extract_patch(centroid_x, centroid_y)
        
        # Normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Convert to torch tensor (H, W, C) -> (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1)
        
        return patch, tubule_id
    
    def _extract_patch(self, center_x, center_y):
        """Extract patch centered on given coordinates"""
        half_size = self.patch_size // 2
        h, w = self.he_image.shape[:2]
        
        # Calculate boundaries
        y_min = center_y - half_size
        y_max = center_y + half_size
        x_min = center_x - half_size
        x_max = center_x + half_size
        
        # Handle boundary cases with padding
        if y_min < 0 or y_max > h or x_min < 0 or x_max > w:
            # Need padding
            patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=self.he_image.dtype)
            
            # Calculate valid regions
            src_y_min = max(0, y_min)
            src_y_max = min(h, y_max)
            src_x_min = max(0, x_min)
            src_x_max = min(w, x_max)
            
            dst_y_min = src_y_min - y_min
            dst_y_max = dst_y_min + (src_y_max - src_y_min)
            dst_x_min = src_x_min - x_min
            dst_x_max = dst_x_min + (src_x_max - src_x_min)
            
            patch[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = \
                self.he_image[src_y_min:src_y_max, src_x_min:src_x_max]
            
            return patch
        else:
            return self.he_image[y_min:y_max, x_min:x_max]

# INFERENCE FUNCTIONS

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load trained fine-tuned model from checkpoint
    
    Args:
        checkpoint_path: Path to .ckpt file
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
        idx_to_label: Dictionary mapping class indices to labels
    """
    print(f"\nLoading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hparams = checkpoint['hyper_parameters']
    num_classes = hparams['num_classes']
    dropout = hparams.get('dropout', 0.3)
    
    print(f"  Model hyperparameters:")
    print(f"    Number of classes: {num_classes}")
    print(f"    Dropout: {dropout}")
    
    # Reconstruct the model architecture manually
    # We need to create a simple inference model that doesn't need Lightning
    class InferenceModel(nn.Module):
        """Simplified model for inference"""
        
        def __init__(self, num_classes, embedding_size=64, num_filters=64, 
                     num_layers=4, dropout=0.3):
            super().__init__()
            self.num_classes = num_classes
            
            # H&E initial conv (from SimCLR model)
            self.he_conv = nn.Sequential(
                nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            
            # Encoder (from SimCLR model)
            self.encoder = CNNEncoder(
                input_channels=num_filters,
                input_size=64,
                num_layers=num_layers,
                num_filters=num_filters,
                embedding_size=embedding_size,
                batch_norm=True
            )
            
            # Classifier head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(embedding_size, num_classes)
            )
        
        def forward(self, he_img):
            # Get H&E embedding
            he_feat = self.he_conv(he_img)
            he_embed = self.encoder(he_feat)
            
            # Classify
            logits = self.classifier(he_embed)
            return logits
    
    # Create model
    model = InferenceModel(
        num_classes=num_classes,
        embedding_size=64,  # Default from training
        num_filters=64,      # Default from training
        num_layers=4,        # Default from training
        dropout=dropout
    )
    
    # Load weights from checkpoint
    state_dict = checkpoint['state_dict']
    
    # Create a new state dict with correct keys for our inference model
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'model.' prefix if present (from Lightning module)
        if key.startswith('model.'):
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = value
        # Keep classifier weights as is
        elif key.startswith('classifier.'):
            new_state_dict[key] = value
    
    # Load the weights
    model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Number of classes: {model.num_classes}")
    
    # Get label mapping (from checkpoint if available, otherwise infer)
    if 'idx_to_label' in checkpoint:
        idx_to_label = checkpoint['idx_to_label']
    else:
        # Default mapping (will be overridden if we can infer from data)
        idx_to_label = {i: f'Class_{i}' for i in range(model.num_classes)}
    
    return model, idx_to_label


def predict_tubules(model, he_image, tubule_mask, patch_size=64, 
                    batch_size=32, num_workers=4, device='cuda'):
    """
    Predict classification for all tubules in image
    
    Args:
        model: Trained model
        he_image: H&E image array
        tubule_mask: Segmentation mask with tubule IDs
        patch_size: Size of patches
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        device: 'cuda' or 'cpu'
    
    Returns:
        predictions: Dictionary mapping tubule_id -> predicted_class_idx
        probabilities: Dictionary mapping tubule_id -> class_probabilities
    """
    
    print("\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    
    # Create dataset
    dataset = WholeSlideInferenceDataset(he_image, tubule_mask, patch_size)
    
    print(f"Found {len(dataset)} tubules to classify")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Run inference
    predictions = {}
    probabilities = {}
    
    model.eval()
    with torch.no_grad():
        for patches, tubule_ids in tqdm(dataloader, desc="Classifying tubules"):
            patches = patches.to(device)
            
            # Get predictions
            logits = model(patches)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Store results
            for i, tid in enumerate(tubule_ids.cpu().numpy()):
                predictions[int(tid)] = int(preds[i].cpu().item())
                probabilities[int(tid)] = probs[i].cpu().numpy()
    
    print(f"\n✓ Classified {len(predictions)} tubules")
    
    return predictions, probabilities


def update_metadata_with_predictions(metadata_file, sample_name, predictions, 
                                     idx_to_label, output_csv):
    """
    Update metadata CSV with predictions
    
    Args:
        metadata_file: Path to metadata CSV
        sample_name: Sample name to filter (e.g., 'LuN_A3_Area4')
        predictions: Dictionary mapping tubule_id -> predicted_class_idx
        idx_to_label: Dictionary mapping class_idx -> label
        output_csv: Path to save updated CSV
    
    Returns:
        updated_df: Updated dataframe
    """
    
    print("\n" + "="*70)
    print("UPDATING METADATA")
    print("="*70)
    
    # Load metadata
    df = pd.read_csv(metadata_file)
    print(f"Loaded metadata: {len(df)} rows")
    
    # Filter for this sample
    sample_df = df[df['Sample_Area'] == sample_name].copy()
    print(f"Found {len(sample_df)} tubules for sample '{sample_name}'")
    
    if len(sample_df) == 0:
        print(f"⚠️  WARNING: No tubules found for sample '{sample_name}'")
        print(f"   Available samples: {df['Sample_Area'].unique()}")
        return df
    
    # Add prediction column
    df['pred_class_idx'] = -1  # Initialize with -1 (not predicted)
    df['pred_class_label'] = 'NotPredicted'
    df['pred_confidence'] = 0.0
    
    # Update predictions for this sample
    matched = 0
    for idx, row in sample_df.iterrows():
        tubule_id = int(row['TubuleID'])
        
        if tubule_id in predictions:
            pred_idx = predictions[tubule_id]
            pred_label = idx_to_label.get(pred_idx, f'Class_{pred_idx}')
            
            df.at[idx, 'pred_class_idx'] = pred_idx
            df.at[idx, 'pred_class_label'] = pred_label
            
            matched += 1
    
    print(f"\n✓ Matched {matched}/{len(sample_df)} tubules with predictions")
    
    # Print prediction distribution
    print("\nPrediction distribution for this sample:")
    pred_counts = df[df['Sample_Area'] == sample_name]['pred_class_label'].value_counts()
    for label, count in pred_counts.items():
        if label != 'NotPredicted':
            print(f"  {label:20s}: {count:4d} ({count/len(sample_df)*100:5.1f}%)")
    
    # Save updated CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved updated metadata to: {output_csv}")
    
    return df


def visualize_predictions(tubule_mask, predictions, idx_to_label, output_path):
    """
    Create visualization of predictions overlaid on mask
    
    Args:
        tubule_mask: Segmentation mask
        predictions: Dictionary mapping tubule_id -> class_idx
        idx_to_label: Dictionary mapping class_idx -> label
        output_path: Path to save visualization
    """
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    # Get unique classes
    unique_classes = sorted(set(predictions.values()))
    num_classes = len(unique_classes)
    
    # Create color map
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    # Create RGB visualization
    h, w = tubule_mask.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color each tubule by its prediction
    for tubule_id, pred_idx in tqdm(predictions.items(), desc="Creating visualization"):
        mask = tubule_mask == tubule_id
        color = colors(unique_classes.index(pred_idx))[:3]
        viz[mask] = (np.array(color) * 255).astype(np.uint8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(viz)
    ax.axis('off')
    ax.set_title('Tubule Classification Predictions', fontsize=20, fontweight='bold')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = []
    for i, class_idx in enumerate(unique_classes):
        label = idx_to_label.get(class_idx, f'Class_{class_idx}')
        count = sum(1 for p in predictions.values() if p == class_idx)
        color = colors(i)[:3]
        legend_elements.append(Patch(facecolor=color, label=f'{label} (n={count})'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             framealpha=0.9, title='Classification', title_fontsize=14)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    plt.show()

# MAIN EXECUTION

def main(config):
    print("\n" + "="*70)
    print("🔬 TUBULE CLASSIFICATION - MODEL DEPLOYMENT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {config['checkpoint_path']}")
    print(f"  H&E Image: {config['he_image_path']}")
    print(f"  Mask: {config['tubule_mask_path']}")
    print(f"  Sample: {config['sample_name']}")
    print(f"  Device: {config['device']}")
    print("="*70)
    
    # Verify files exist
    if not os.path.exists(config['checkpoint_path']):
        print(f"\n❌ ERROR: Checkpoint not found: {config['checkpoint_path']}")
        return
    
    if not os.path.exists(config['he_image_path']):
        print(f"\n❌ ERROR: H&E image not found: {config['he_image_path']}")
        return
    
    if not os.path.exists(config['tubule_mask_path']):
        print(f"\n❌ ERROR: Tubule mask not found: {config['tubule_mask_path']}")
        return
    
    if not os.path.exists(config['metadata_file']):
        print(f"\n❌ ERROR: Metadata file not found: {config['metadata_file']}")
        return
    
    # Load model
    model, idx_to_label = load_trained_model(
        config['checkpoint_path'],
        config['device']
    )
    
    # Infer label mapping from metadata if available
    try:
        metadata = pd.read_csv(config['metadata_file'])
        if 'State' in metadata.columns:
            unique_labels = sorted(metadata['State'].unique())
            idx_to_label = {i: label for i, label in enumerate(unique_labels)}
            print(f"\n✓ Inferred class labels from metadata:")
            for idx, label in idx_to_label.items():
                print(f"    {idx}: {label}")
    except Exception as e:
        print(f"\n⚠️  Could not infer labels from metadata: {e}")
    
    # Load images
    print("\n" + "="*70)
    print("LOADING IMAGES")
    print("="*70)
    
    print(f"Loading H&E image...")
    he_image = imread(config['he_image_path'])
    print(f"  Shape: {he_image.shape}")
    print(f"  Dtype: {he_image.dtype}")
    
    print(f"\nLoading tubule mask...")
    tubule_mask = imread(config['tubule_mask_path'])
    print(f"  Shape: {tubule_mask.shape}")
    print(f"  Dtype: {tubule_mask.dtype}")
    print(f"  Unique tubules: {len(np.unique(tubule_mask)) - 1}")  # -1 for background
    
    # Run inference
    predictions, probabilities = predict_tubules(
        model,
        he_image,
        tubule_mask,
        config['patch_size'],
        config['batch_size'],
        config['num_workers'],
        config['device']
    )
    
    # Print prediction summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    pred_counts = {}
    for pred_idx in predictions.values():
        label = idx_to_label.get(pred_idx, f'Class_{pred_idx}')
        pred_counts[label] = pred_counts.get(label, 0) + 1
    
    for label, count in sorted(pred_counts.items()):
        print(f"  {label:20s}: {count:4d} ({count/len(predictions)*100:5.1f}%)")
    
    # Update metadata
    updated_df = update_metadata_with_predictions(
        config['metadata_file'],
        config['sample_name'],
        predictions,
        idx_to_label,
        config['output_csv']
    )
    
    # Create visualization
    if config['save_visualization']:
        visualize_predictions(
            tubule_mask,
            predictions,
            idx_to_label,
            config['viz_output']
        )
    
    print("\n" + "="*70)
    print("✓ DEPLOYMENT COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  Updated CSV: {config['output_csv']}")
    if config['save_visualization']:
        print(f"  Visualization: {config['viz_output']}")
    print("\nYou can now:")
    print("  - Check the updated CSV for predictions")
    print("  - View the visualization to see spatial distribution")
    print("  - Analyze prediction confidence scores")
    print("="*70)
    
    return updated_df, predictions, probabilities


if __name__ == "__main__":
    try:
        updated_df, predictions, probabilities = main(CONFIG)
        
        print("\n" + "="*70)
        print("VARIABLES AVAILABLE IN WORKSPACE")
        print("="*70)
        print("  updated_df     - DataFrame with predictions")
        print("  predictions    - Dict: tubule_id -> class_idx")
        print("  probabilities  - Dict: tubule_id -> class_probs")
        print("\nYou can now explore these in Spyder's Variable Explorer!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()