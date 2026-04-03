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
import re
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tifffile import imread
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

CONFIG = {
    # ===== Model Checkpoint =====
    'checkpoint_path': '/home/thaocao/Palom/logs/checkpoints/finetune-epoch=79-val_acc=0.9974.ckpt',
    
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
    'viz_output': './predictions/visualization_11042025_states.tif',
    
    # ===== Device =====
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ============================================================================
# MODEL ARCHITECTURE - Contrastive Learning Model (trained on H&E and CODEX)
# ============================================================================

class CNNEncoder(nn.Module):
    """CNN encoder for tubule patches - used in contrastive learning framework"""
    
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
    """
    Complete SimCLR contrastive learning model with encoders and projection head.
    Trained on paired H&E (64x64x3) and CODEX (64x64x4) patches.
    For deployment, we use only the H&E branch for inference.
    """
    
    def __init__(self, input_channels=3, input_size=64, num_layers=4,
                 num_filters=64, embedding_size=64, projection_dim=64,
                 batch_norm=True):
        super(SimCLRModel, self).__init__()
        
        # Separate initial conv for H&E (3 channels) and CODEX (4 channels)
        self.he_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        # CODEX branch (not used during H&E-only inference, but part of trained model)
        self.codex_conv = nn.Sequential(
            nn.Conv2d(4, num_filters, kernel_size=3, padding=1),  # 4 channels for CODEX
            nn.LeakyReLU()
        )
        
        # Shared encoder (used by both H&E and CODEX during training)
        self.encoder = CNNEncoder(
            input_channels=num_filters,
            input_size=input_size,
            num_layers=num_layers,
            num_filters=num_filters,
            embedding_size=embedding_size,
            batch_norm=batch_norm
        )
        
        # Projection head (used during contrastive training)
        self.projection = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, projection_dim)
        )
    
    def forward(self, he_img, codex_img=None):
        """
        Forward pass. During inference, only H&E images are provided.
        
        Args:
            he_img: H&E image tensor (B, 3, H, W)
            codex_img: CODEX image tensor (B, 4, H, W) - optional, only used during training
        """
        # Process H&E (always available)
        he_feat = self.he_conv(he_img)
        he_embed = self.encoder(he_feat)
        he_proj = self.projection(he_embed)
        
        if codex_img is not None:
            # Process CODEX (only during training with paired data)
            codex_feat = self.codex_conv(codex_img)
            codex_embed = self.encoder(codex_feat)
            codex_proj = self.projection(codex_embed)
            return he_embed, codex_embed, he_proj, codex_proj
        
        return he_embed, he_proj


# ============================================================================
# DATASET FOR WHOLE SLIDE INFERENCE
# ============================================================================

class WholeSlideInferenceDataset(Dataset):
    """
    Dataset for extracting patches from whole slide H&E image.
    Ensures all tubule instances in the tubule_mask are processed.
    """
    
    def __init__(self, he_image, tubule_mask, patch_size=64):
        """
        Args:
            he_image: H&E image array (H, W, C) or (C, H, W)
            tubule_mask: Segmentation mask where each tubule has unique ID
            patch_size: Size of patches to extract (64x64 to match training)
        """
        self.he_image = he_image
        self.tubule_mask = tubule_mask
        self.patch_size = patch_size
        
        # Ensure correct shape (H, W, C) for H&E
        if self.he_image.shape[0] == 3 or self.he_image.shape[0] == 4:
            self.he_image = np.transpose(self.he_image, (1, 2, 0))
        
        # Get list of ALL tubule IDs from mask (ensuring all instances are included)
        self.tubule_ids = np.unique(tubule_mask)
        self.tubule_ids = self.tubule_ids[self.tubule_ids > 0]  # Exclude background
        self.tubule_ids = self.tubule_ids.astype(np.int32)
        
        print(f"  Found {len(self.tubule_ids)} unique tubule instances in mask")
        
        # Calculate centroids for each tubule instance
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
        
        # Extract 64x64 patch centered on tubule (matching training patch size)
        patch = self._extract_patch(centroid_x, centroid_y)
        
        # Normalize to [0, 1]
        patch = patch.astype(np.float32) / 255.0
        
        # Convert to torch tensor (H, W, C) -> (C, H, W)
        patch = torch.from_numpy(patch).permute(2, 0, 1)
        
        return patch, tubule_id
    
    def _extract_patch(self, center_x, center_y):
        """Extract 64x64 patch centered on given coordinates with padding if needed"""
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


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load trained fine-tuned contrastive learning model from checkpoint.
    The model was trained on paired H&E and CODEX patches, but for deployment
    we only use the H&E branch.
    
    Args:
        checkpoint_path: Path to .ckpt file
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
        idx_to_label: Dictionary mapping class indices to labels
    """
    print(f"\nLoading contrastive learning model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    hparams = checkpoint['hyper_parameters']
    num_classes = hparams['num_classes']
    dropout = hparams.get('dropout', 0.3)
    
    # Load state dict to infer architecture
    state_dict = checkpoint['state_dict']
    
    # Infer embedding_size from classifier weights
    classifier_key = None
    for key in state_dict.keys():
        if 'classifier' in key and 'weight' in key and key.endswith('weight'):
            classifier_key = key
            break
    
    if classifier_key:
        # Shape is (num_classes, embedding_size)
        embedding_size = state_dict[classifier_key].shape[1]
    else:
        embedding_size = 64  # Default fallback
    
    # Infer num_filters from he_conv weights
    he_conv_key = None
    for key in state_dict.keys():
        if 'he_conv' in key and 'weight' in key:
            he_conv_key = key
            break
    
    if he_conv_key:
        # Shape is (out_channels, in_channels, kernel, kernel)
        num_filters = state_dict[he_conv_key].shape[0]
    else:
        num_filters = 64  # Default fallback
    
    # Infer num_layers by counting the conv layers in encoder
    num_layers = 0
    for key in state_dict.keys():
        if 'encoder.cnn' in key and '.0.weight' in key:
            # Count unique layer indices
            match = re.search(r'encoder\.cnn\.(\d+)\.', key)
            if match:
                layer_idx = int(match.group(1))
                # Layers are: conv(0), relu(1), bn(2), pool(3), conv(4), ...
                # So we have a new conv layer every 4 indices
                num_layers = max(num_layers, (layer_idx // 4) + 1)
    
    if num_layers == 0:
        num_layers = 4  # Default fallback
    
    print(f"  Inferred architecture from checkpoint:")
    print(f"    Number of classes: {num_classes}")
    print(f"    Embedding size: {embedding_size}")
    print(f"    Initial filters: {num_filters}")
    print(f"    Number of layers: {num_layers}")
    print(f"    Dropout: {dropout}")
    print(f"  Note: Model trained with contrastive learning on H&E + CODEX")
    print(f"        Deploying H&E branch only for inference")
    
    # Reconstruct the model architecture for inference
    class InferenceModel(nn.Module):
        """Simplified model for H&E-only inference from contrastive learning model"""
        
        def __init__(self, num_classes, embedding_size=64, num_filters=64, 
                     num_layers=4, dropout=0.3):
            super().__init__()
            self.num_classes = num_classes
            
            # H&E initial conv (from SimCLR model's H&E branch)
            self.he_conv = nn.Sequential(
                nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            
            # Encoder (shared encoder from contrastive learning)
            self.encoder = CNNEncoder(
                input_channels=num_filters,
                input_size=64,
                num_layers=num_layers,
                num_filters=num_filters,
                embedding_size=embedding_size,
                batch_norm=True
            )
            
            # Classifier head (fine-tuned on labeled data)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(embedding_size, num_classes)
            )
        
        def forward(self, he_img):
            # Get H&E embedding using contrastive-learned features
            he_feat = self.he_conv(he_img)
            he_embed = self.encoder(he_feat)
            
            # Classify
            logits = self.classifier(he_embed)
            return logits
    
    # Create model with inferred architecture
    model = InferenceModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        num_filters=num_filters,
        num_layers=num_layers,
        dropout=dropout
    )
    
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
        else:
            new_state_dict[key] = value
    
    # Load the weights
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ Model loaded successfully")
    except RuntimeError as e:
        print(f"⚠️  Warning during model loading: {e}")
        print(f"   Attempting to load with filtered parameters...")
        # Try to load only matching keys
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in new_state_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(filtered_dict, strict=False)
        print(f"✓ Loaded {len(filtered_dict)}/{len(model_dict)} matching parameters")
    
    model.to(device)
    model.eval()
    
    print(f"  Model ready for inference")
    print(f"  Number of classes: {model.num_classes}")
    
    # Get label mapping
    if 'idx_to_label' in checkpoint:
        idx_to_label = checkpoint['idx_to_label']
    else:
        # Default mapping
        idx_to_label = {i: f'Class_{i}' for i in range(model.num_classes)}
    
    return model, idx_to_label


def predict_tubules(model, he_image, tubule_mask, patch_size=64, 
                    batch_size=32, num_workers=4, device='cuda'):
    """
    Predict classification for ALL tubule instances in the mask.
    
    Args:
        model: Trained contrastive learning model
        he_image: H&E image array
        tubule_mask: Segmentation mask with tubule IDs
        patch_size: Size of patches (64x64 to match training)
        batch_size: Batch size for inference
        num_workers: DataLoader workers
        device: 'cuda' or 'cpu'
    
    Returns:
        predictions: Dictionary mapping tubule_id -> predicted_class_idx
        probabilities: Dictionary mapping tubule_id -> class_probabilities
    """
    
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON ALL TUBULE INSTANCES")
    print("="*70)
    
    # Create dataset (ensures all tubule instances are included)
    dataset = WholeSlideInferenceDataset(he_image, tubule_mask, patch_size)
    
    print(f"Classifying {len(dataset)} tubule instances")
    
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
    
    print(f"\n✓ Successfully classified all {len(predictions)} tubule instances")
    
    return predictions, probabilities


def update_metadata_with_predictions(metadata_file, sample_name, predictions, 
                                     idx_to_label, output_csv):
    """
    Create new dataframe with only the sample's rows, add predictions,
    and calculate precision, recall, and F1 scores.
    
    Args:
        metadata_file: Path to metadata CSV
        sample_name: Sample name to filter (e.g., 'LuN_012523S5_Area2')
        predictions: Dictionary mapping tubule_id -> predicted_class_idx
        idx_to_label: Dictionary mapping class_idx -> label
        output_csv: Path to save updated CSV
    
    Returns:
        sample_df: New dataframe with only this sample's data and predictions
        metrics_dict: Dictionary containing precision, recall, F1 scores
    """
    
    print("\n" + "="*70)
    print("UPDATING METADATA WITH PREDICTIONS")
    print("="*70)
    
    # Load full metadata
    df_full = pd.read_csv(metadata_file)
    print(f"Loaded full metadata: {len(df_full)} rows")
    
    # Filter to create NEW dataframe with only this sample's rows
    sample_df = df_full[df_full['Sample_Area'] == sample_name].copy()
    print(f"Filtered to {len(sample_df)} rows for sample '{sample_name}'")
    
    if len(sample_df) == 0:
        print(f"⚠️  WARNING: No tubules found for sample '{sample_name}'")
        print(f"   Available samples: {df_full['Sample_Area'].unique()}")
        return sample_df, {}
    
    # Add new columns for contrastive model predictions
    sample_df['ContrastiveModel_PredictedClass_Idx'] = -1
    sample_df['ContrastiveModel_PredictedState'] = 'NotPredicted'
    
    # Update predictions
    matched = 0
    for idx, row in sample_df.iterrows():
        tubule_id = int(row['TubuleID'])
        
        if tubule_id in predictions:
            pred_idx = predictions[tubule_id]
            pred_label = idx_to_label.get(pred_idx, f'Class_{pred_idx}')
            
            sample_df.at[idx, 'ContrastiveModel_PredictedClass_Idx'] = pred_idx
            sample_df.at[idx, 'ContrastiveModel_PredictedState'] = pred_label
            matched += 1
    
    print(f"✓ Matched {matched}/{len(sample_df)} tubules with predictions")
    
    # Calculate metrics if ground truth 'State' column exists
    metrics_dict = {}
    
    if 'State' in sample_df.columns:
        print("\n" + "="*70)
        print("CALCULATING METRICS (Ground Truth vs Predicted)")
        print("="*70)
        
        # Filter to only rows with predictions
        eval_df = sample_df[sample_df['ContrastiveModel_PredictedState'] != 'NotPredicted'].copy()
        
        if len(eval_df) > 0:
            y_true = eval_df['State'].values
            y_pred = eval_df['ContrastiveModel_PredictedState'].values
            
            # Get unique labels
            labels = sorted(list(set(y_true) | set(y_pred)))
            
            # Calculate metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=None, zero_division=0
            )
            
            # Calculate macro averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            # Calculate weighted averages
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Store metrics
            metrics_dict = {
                'labels': labels,
                'precision_per_class': precision,
                'recall_per_class': recall,
                'f1_per_class': f1,
                'support_per_class': support,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'num_samples': len(eval_df)
            }
            
            # Print per-class metrics
            print("\nPer-Class Metrics:")
            print(f"{'State':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 70)
            for i, label in enumerate(labels):
                print(f"{label:<25} {precision[i]:>11.4f} {recall[i]:>11.4f} {f1[i]:>11.4f} {support[i]:>9}")
            
            print("\n" + "-" * 70)
            print(f"{'Macro Average':<25} {precision_macro:>11.4f} {recall_macro:>11.4f} {f1_macro:>11.4f} {len(eval_df):>9}")
            print(f"{'Weighted Average':<25} {precision_weighted:>11.4f} {recall_weighted:>11.4f} {f1_weighted:>11.4f} {len(eval_df):>9}")
            
            # Print confusion matrix
            print("\n" + "="*70)
            print("CONFUSION MATRIX")
            print("="*70)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            # Print with labels
            print(f"\n{'True Pred':<20}", end="")
            for label in labels:
                print(f"{label[:15]:>15}", end="")
            print()
            print("-" * (20 + 15 * len(labels)))
            
            for i, true_label in enumerate(labels):
                print(f"{true_label[:20]:<20}", end="")
                for j in range(len(labels)):
                    print(f"{cm[i, j]:>15}", end="")
                print()
            
            # Print classification report
            print("\n" + "="*70)
            print("DETAILED CLASSIFICATION REPORT")
            print("="*70)
            print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
            
        else:
            print("⚠️  No predictions available for metric calculation")
    else:
        print("\n⚠️  'State' column not found in metadata - skipping metric calculation")
    
    # Print prediction distribution
    print("\n" + "="*70)
    print("PREDICTION DISTRIBUTION")
    print("="*70)
    pred_counts = sample_df['ContrastiveModel_PredictedState'].value_counts()
    for label, count in pred_counts.items():
        if label != 'NotPredicted':
            print(f"  {label:<30}: {count:>4d} ({count/len(sample_df)*100:>5.1f}%)")
    
    # Save the new dataframe (only this sample's data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sample_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved predictions for sample '{sample_name}' to: {output_csv}")
    
    return sample_df, metrics_dict


def visualize_predictions(tubule_mask, predictions, idx_to_label, output_path):
    """
    Create visualization of predictions overlaid on mask using state-specific colors.
    
    Args:
        tubule_mask: Segmentation mask
        predictions: Dictionary mapping tubule_id -> class_idx
        idx_to_label: Dictionary mapping class_idx -> label
        output_path: Path to save visualization
    """
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    # Define colors for each tubule state
    state_colors = {
        'Healthy': (184, 225, 134),
        'Stressed': (241, 182, 218),
        'Inflamed': (253, 184, 99),
        'Stressed and Inflamed': (202, 0, 32),
        'Atrophic': (150, 100, 100)
    }
    
    # Create RGB visualization
    h, w = tubule_mask.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color each tubule by its predicted state
    for tubule_id, pred_idx in tqdm(predictions.items(), desc="Creating visualization"):
        mask = tubule_mask == tubule_id
        pred_label = idx_to_label.get(pred_idx, f'Class_{pred_idx}')
        
        # Get color for this state
        if pred_label in state_colors:
            color = state_colors[pred_label]
        else:
            # Default color for unknown states
            color = (128, 128, 128)
        
        viz[mask] = color
    
    # Create figure without legend
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(viz)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Also save as TIFF for further analysis
    from tifffile import imwrite
    tiff_path = output_path.replace('.tif', '_rgb.tif')
    imwrite(tiff_path, viz)
    print(f"✓ Saved TIFF visualization to: {tiff_path}")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(config):
    print("\n" + "="*70)
    print("🔬 TUBULE CLASSIFICATION - CONTRASTIVE MODEL DEPLOYMENT")
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
        return None, None, None, None
    
    if not os.path.exists(config['he_image_path']):
        print(f"\n❌ ERROR: H&E image not found: {config['he_image_path']}")
        return None, None, None, None
    
    if not os.path.exists(config['tubule_mask_path']):
        print(f"\n❌ ERROR: Tubule mask not found: {config['tubule_mask_path']}")
        return None, None, None, None
    
    if not os.path.exists(config['metadata_file']):
        print(f"\n❌ ERROR: Metadata file not found: {config['metadata_file']}")
        return None, None, None, None
    
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
    
    # Run inference on ALL tubule instances
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
        print(f"  {label:<30}: {count:>4d} ({count/len(predictions)*100:>5.1f}%)")
    
    # Update metadata and calculate metrics
    sample_df, metrics_dict = update_metadata_with_predictions(
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
    print(f"  Predictions CSV: {config['output_csv']}")
    if config['save_visualization']:
        print(f"  Visualization: {config['viz_output']}")
    
    if metrics_dict:
        print(f"\nPerformance Summary:")
        print(f"  Precision (macro):  {metrics_dict['precision_macro']:.4f}")
        print(f"  Recall (macro):     {metrics_dict['recall_macro']:.4f}")
        print(f"  F1-Score (macro):   {metrics_dict['f1_macro']:.4f}")
        print(f"  Samples evaluated:  {metrics_dict['num_samples']}")
    
    print("\nYou can now:")
    print("  - Review the predictions CSV with ground truth comparison")
    print("  - Examine per-class precision, recall, and F1 scores")
    print("  - View the visualization with state-specific colors")
    print("  - Analyze the confusion matrix for error patterns")
    print("="*70)
    
    return sample_df, predictions, probabilities, metrics_dict


if __name__ == "__main__":
    try:
        sample_df, predictions, probabilities, metrics = main(CONFIG)
        
        print("\n" + "="*70)
        print("VARIABLES AVAILABLE IN WORKSPACE")
        print("="*70)
        print("  sample_df      - DataFrame with predictions (sample only)")
        print("  predictions    - Dict: tubule_id -> class_idx")
        print("  probabilities  - Dict: tubule_id -> class_probs")
        print("  metrics        - Dict: precision, recall, F1 scores")
        print("\nYou can now explore these in Spyder's Variable Explorer!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()