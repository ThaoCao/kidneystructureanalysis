#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:32:14 2025

@author: thaocao
"""

from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import os
import random
import matplotlib.pyplot as plt
from skimage import measure

np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

Image.MAX_IMAGE_PIXELS = None

# Configuration
CONFIG = {
    # Root directory for patch data
    'patch_dir': '/home/thaocao/Palom/data/Lupus_Nephritis',
    
    # Metadata CSV with State labels
    'metadata_csv': '/home/thaocao/structureanalysis/tubules_classified_IDs_with_distances2_centroids.csv',
    
    # Output CSV for reviewed labels
    'output_csv': '/home/thaocao/Palom/data/Lupus_Nephritis/GUI/reviewed_tubule_labels.csv',
    
    # Tracking files
    'reviewed_txt': '/home/thaocao/Palom/data/Lupus_Nephritis/GUI/reviewed.txt',
    'skipped_txt': '/home/thaocao/Palom/data/Lupus_Nephritis/GUI/skipped.txt',
    
    # Patch size
    'patch_size': 64,
    
    # CODEX channel names (order matters - should match extraction script)
    'codex_channels': ['CD10', 'Claudin1', 'MUC1', 'CD45'],
    
    # Process random order?
    'shuffle': True,
}

# UI Colors
DARK_BG = "#1e1e1e"
DARK_FG = "#f0f0f0"
BUTTON_BG = "#363636"
BUTTON_HOVER = "#404040"
BUTTON_ACTIVE = "#2d2d2d"
ACCENT_COLOR = "#007acc"
ACCEPT_COLOR = "#4CAF50"
REJECT_COLOR = "#F44336"
DARK_FRAME = "#3c3c3c"
BUTTON_FG = "#ffffff"
SCROLLBAR_BG = "#404040"

# Colormap LUT
_VIRIDIS_LUT = (plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
_MAGMA_LUT = (plt.get_cmap("magma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

terminate = False


def load_tracked_files(file_path):
    """Load set of already processed files."""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def get_unique_states(metadata_df):
    """Get all unique State values from metadata."""
    if 'State' in metadata_df.columns:
        states = metadata_df['State'].unique()
        return sorted([s for s in states if pd.notna(s)])
    return []


def parse_filename(filename):
    """
    Parse tubule patch filename to extract metadata.
    Format: LuN_A3_Area4_TID1523_CNormal_SHealthy_X2450_Y3891.npy
    """
    fname = filename.replace('.npy', '')
    parts = fname.split('_')
    
    info = {'filename': filename}
    
    try:
        # Extract sample area (e.g., LuN_A3_Area4)
        sample_parts = []
        for i, p in enumerate(parts):
            if p.startswith('TID'):
                sample_parts = parts[:i]
                break
        info['sample_area'] = '_'.join(sample_parts)
        
        # Extract tubule ID
        tid_parts = [p for p in parts if p.startswith('TID')]
        if tid_parts:
            info['tubule_id'] = int(tid_parts[0].replace('TID', ''))
        
        # Extract classification
        class_parts = [p for p in parts if p.startswith('C') and not p.startswith('Centroid')]
        if class_parts:
            info['classification'] = class_parts[0][1:]  # Remove 'C' prefix
        
        # Extract state
        state_parts = [p for p in parts if p.startswith('S')]
        if state_parts:
            info['state'] = state_parts[0][1:]  # Remove 'S' prefix
        
        # Extract centroid
        x_parts = [p for p in parts if p.startswith('X')]
        y_parts = [p for p in parts if p.startswith('Y')]
        if x_parts:
            info['centroid_x'] = int(x_parts[0][1:])
        if y_parts:
            info['centroid_y'] = int(y_parts[0][1:])
            
    except Exception as e:
        print(f"Warning: Could not fully parse filename {filename}: {e}")
    
    return info


def normalize_to_uint8(img, percentile=99.9):
    """Normalize image to uint8 range."""
    img = img.astype(np.float32)
    vmax = float(np.percentile(img, percentile))
    if vmax > 0:
        img = img / vmax
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def apply_colormap(img_u8, colormap='viridis'):
    """Apply colormap to grayscale image."""
    if colormap == 'magma':
        return _MAGMA_LUT[img_u8]
    else:
        return _VIRIDIS_LUT[img_u8]


def draw_mask_contour(rgb_img, mask, color=(0, 255, 0)):
    """Draw contour of mask on RGB image."""
    if np.any(mask):
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = contour.astype(int)
            # Keep in bounds
            contour = contour[(contour[:, 0] >= 0) &
                            (contour[:, 0] < rgb_img.shape[0]) &
                            (contour[:, 1] >= 0) &
                            (contour[:, 1] < rgb_img.shape[1])]
            rgb_img[contour[:, 0], contour[:, 1]] = np.array(color, dtype=np.uint8)
    return rgb_img


def main():
    global terminate
    
    # Setup directories
    patch_dir = CONFIG['patch_dir']
    metadata_csv = CONFIG['metadata_csv']
    output_csv = CONFIG['output_csv']
    reviewed_txt = CONFIG['reviewed_txt']
    skipped_txt = CONFIG['skipped_txt']
    patch_size = CONFIG['patch_size']
    codex_channels = CONFIG['codex_channels']
    
    # Create output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Patch directories
    he_dir = os.path.join(patch_dir, f'he_patches_{patch_size}')
    codex_dir = os.path.join(patch_dir, f'codex_patches_{patch_size}')
    mask_dir = os.path.join(patch_dir, f'tubule_masks_{patch_size}')
    
    # Check directories exist
    for d, name in [(he_dir, 'H&E'), (codex_dir, 'CODEX'), (mask_dir, 'Masks')]:
        if not os.path.exists(d):
            print(f"❌ ERROR: {name} directory not found: {d}")
            return
    
    # Load metadata
    print(f"\nLoading metadata from: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv)
    print(f"✓ Loaded {len(metadata_df)} tubule records")
    
    # Get unique states for selection
    unique_states = get_unique_states(metadata_df)
    print(f"✓ Found {len(unique_states)} unique States: {unique_states}")
    
    # Load tracking files
    reviewed_set = load_tracked_files(reviewed_txt)
    skipped_set = load_tracked_files(skipped_txt)
    
    # Get all patch files
    all_files = [f for f in os.listdir(he_dir) if f.endswith('.npy')]
    
    # Filter to unprocessed files
    valid_files = [f for f in all_files if f not in reviewed_set and f not in skipped_set]
    
    if CONFIG['shuffle']:
        random.shuffle(valid_files)
    
    print(f"\n✓ Found {len(all_files)} patches")
    print(f"  - {len(reviewed_set)} already reviewed")
    print(f"  - {len(skipped_set)} skipped")
    print(f"  - {len(valid_files)} remaining to review\n")
    
    if len(valid_files) == 0:
        print("No patches to review!")
        return
    
    # Initialize or load output CSV
    if os.path.exists(output_csv):
        output_df = pd.read_csv(output_csv)
    else:
        output_df = pd.DataFrame(columns=[
            'filename', 'sample_area', 'tubule_id', 'classification',
            'original_state', 'reviewed_state', 'accepted', 'centroid_x', 'centroid_y'
        ])
    
    # Process each file
    for idx, filename in enumerate(valid_files):
        if terminate:
            print("\n✓ Exiting...")
            return
        
        print(f"\n[{idx+1}/{len(valid_files)}] Processing: {filename}")
        
        # Parse filename
        file_info = parse_filename(filename)
        current_state = file_info.get('state', 'Unknown')
        
        print(f"  Current State: {current_state}")
        
        # Load patches
        try:
            he_patch = np.load(os.path.join(he_dir, filename))
            codex_patch = np.load(os.path.join(codex_dir, filename))
            mask_patch = np.load(os.path.join(mask_dir, filename))
        except Exception as e:
            print(f"  ❌ Error loading patches: {e}")
            with open(skipped_txt, 'a') as f:
                f.write(f'{filename}\n')
            continue
        
        print(f"  Shapes - H&E: {he_patch.shape}, CODEX: {codex_patch.shape}, Mask: {mask_patch.shape}")
        
        # Verify CODEX channels
        n_codex_channels = codex_patch.shape[2] if codex_patch.ndim == 3 else 1
        if n_codex_channels != len(codex_channels):
            print(f"  ⚠️  Warning: Expected {len(codex_channels)} CODEX channels, got {n_codex_channels}")
        
        # === CREATE GUI ===
        root = tk.Tk()
        root.title(f"Tubule Label Review - {filename}")
        root.geometry("1600x900")
        root.configure(bg=DARK_BG)
        root.resizable(True, True)
        
        # Variables to store user response
        user_response = {'accepted': None, 'new_state': None}
        
        # Main layout
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, minsize=250)
        
        # Left: Scrollable image area
        container = tk.Frame(root, bg=DARK_BG)
        container.grid(row=0, column=0, sticky="nsew")
        
        canvas = tk.Canvas(container, highlightthickness=0, bg=DARK_BG)
        vbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview, 
                           bg=SCROLLBAR_BG, troughcolor=DARK_BG, width=12)
        canvas.configure(yscrollcommand=vbar.set)
        
        vbar.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        content = tk.Frame(canvas, bg=DARK_BG)
        win = canvas.create_window((0, 0), window=content, anchor="nw")
        
        def on_content_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", on_content_configure)
        
        def on_canvas_configure(event):
            canvas.itemconfigure(win, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            if event.num == 5 or event.delta < 0:
                canvas.yview_scroll(3, "units")
            elif event.num == 4 or event.delta > 0:
                canvas.yview_scroll(-3, "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_mousewheel)
        canvas.bind_all("<Button-5>", on_mousewheel)
        
        # Right: Control panel
        controls = tk.Frame(root, bg=DARK_BG)
        controls.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        
        # Keep photo references
        photo_refs = []
        
        # === DISPLAY IMAGES ===
        row_idx = 0
        
        # Header: File info
        info_frame = tk.Frame(content, bg=DARK_FRAME, relief="solid", bd=1)
        info_frame.grid(row=row_idx, column=0, columnspan=6, sticky="ew", padx=10, pady=(10, 15))
        
        tk.Label(info_frame, text=f"File: {filename}", font=("Arial", 11, "bold"),
                bg=DARK_FRAME, fg=DARK_FG, anchor="w").pack(fill="x", padx=10, pady=5)
        tk.Label(info_frame, text=f"Sample: {file_info.get('sample_area', 'N/A')}", 
                font=("Arial", 10), bg=DARK_FRAME, fg=DARK_FG, anchor="w").pack(fill="x", padx=10)
        tk.Label(info_frame, text=f"Tubule ID: {file_info.get('tubule_id', 'N/A')}", 
                font=("Arial", 10), bg=DARK_FRAME, fg=DARK_FG, anchor="w").pack(fill="x", padx=10)
        tk.Label(info_frame, text=f"Classification: {file_info.get('classification', 'N/A')}", 
                font=("Arial", 10), bg=DARK_FRAME, fg=DARK_FG, anchor="w").pack(fill="x", padx=10, pady=(0, 5))
        
        row_idx += 1
        
        # Current State label (large, prominent)
        state_frame = tk.Frame(content, bg=ACCENT_COLOR, relief="solid", bd=2)
        state_frame.grid(row=row_idx, column=0, columnspan=6, sticky="ew", padx=10, pady=(0, 20))
        
        tk.Label(state_frame, text="Current State:", font=("Arial", 12, "bold"),
                bg=ACCENT_COLOR, fg=DARK_FG).pack(side="left", padx=10, pady=8)
        tk.Label(state_frame, text=current_state, font=("Arial", 16, "bold"),
                bg=ACCENT_COLOR, fg="white").pack(side="left", padx=5, pady=8)
        
        row_idx += 1
        
        # Display function
        def display_channel(img, title, row, col, colormap='viridis', show_mask=False):
            # Normalize
            img_u8 = normalize_to_uint8(img, percentile=99.9)
            
            # Apply colormap
            rgb = apply_colormap(img_u8, colormap)
            
            # Draw mask contour if requested
            if show_mask and np.any(mask_patch):
                rgb = draw_mask_contour(rgb, mask_patch, color=(0, 255, 0))
            
            # Convert to PIL and create PhotoImage
            pil_img = Image.fromarray(rgb, mode="RGB")
            pil_img = pil_img.resize((200, 200), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            photo_refs.append(photo)
            
            # Title label
            title_label = tk.Label(content, text=title, font=("Arial", 11, "bold"),
                                  bg=DARK_BG, fg=DARK_FG)
            title_label.grid(row=row, column=col, pady=(5, 3), padx=10)
            
            # Image label
            img_label = tk.Label(content, image=photo, bg=DARK_BG, relief="solid", bd=1)
            img_label.image = photo
            img_label.grid(row=row+1, column=col, padx=10, pady=(0, 15))
        
        # Row 1: H&E
        display_channel(he_patch if he_patch.ndim == 2 else he_patch[:, :, 0], 
                       "H&E (Red)", row_idx, 0, colormap='magma', show_mask=False)
        if he_patch.ndim == 3 and he_patch.shape[2] > 1:
            display_channel(he_patch[:, :, 1], "H&E (Green)", row_idx, 1, colormap='magma', show_mask=False)
        if he_patch.ndim == 3 and he_patch.shape[2] > 2:
            display_channel(he_patch[:, :, 2], "H&E (Blue)", row_idx, 2, colormap='magma', show_mask=False)
        
        # H&E composite
        if he_patch.ndim == 3:
            he_rgb = he_patch.astype(np.uint8) if he_patch.dtype == np.uint8 else \
                     normalize_to_uint8(he_patch.mean(axis=2))
            if he_patch.shape[2] == 3:
                he_rgb = he_patch.astype(np.uint8)
            else:
                he_rgb = np.stack([he_patch[:,:,i] for i in range(min(3, he_patch.shape[2]))], axis=-1)
                he_rgb = he_rgb.astype(np.uint8) if he_rgb.dtype == np.uint8 else \
                         (normalize_to_uint8(he_rgb[:,:,0]), normalize_to_uint8(he_rgb[:,:,1]), 
                          normalize_to_uint8(he_rgb[:,:,2]))
                he_rgb = np.stack([normalize_to_uint8(he_rgb[:,:,i]) for i in range(3)], axis=-1)
            
            he_rgb = draw_mask_contour(he_rgb, mask_patch, color=(0, 255, 0))
            pil_he = Image.fromarray(he_rgb, mode="RGB")
            pil_he = pil_he.resize((200, 200), Image.Resampling.LANCZOS)
            photo_he = ImageTk.PhotoImage(pil_he)
            photo_refs.append(photo_he)
            
            tk.Label(content, text="H&E Composite", font=("Arial", 11, "bold"),
                    bg=DARK_BG, fg=DARK_FG).grid(row=row_idx, column=3, pady=(5, 3), padx=10)
            img_label = tk.Label(content, image=photo_he, bg=DARK_BG, relief="solid", bd=1)
            img_label.image = photo_he
            img_label.grid(row=row_idx+1, column=3, padx=10, pady=(0, 15))
        
        row_idx += 2
        
        # Row 2: CODEX channels
        for ch_idx in range(min(len(codex_channels), n_codex_channels)):
            col = ch_idx % 4
            if ch_idx > 0 and col == 0:
                row_idx += 2
            
            channel_data = codex_patch[:, :, ch_idx] if codex_patch.ndim == 3 else codex_patch
            channel_name = codex_channels[ch_idx] if ch_idx < len(codex_channels) else f"Ch{ch_idx}"
            
            show_mask = (ch_idx == 0)  # Show mask on first CODEX channel
            display_channel(channel_data, f"CODEX - {channel_name}", 
                          row_idx, col, colormap='viridis', show_mask=show_mask)
        
        row_idx += 2
        
        # Row 3: Mask
        mask_u8 = (mask_patch * 255).astype(np.uint8)
        mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
        pil_mask = Image.fromarray(mask_rgb, mode="RGB")
        pil_mask = pil_mask.resize((200, 200), Image.Resampling.NEAREST)
        photo_mask = ImageTk.PhotoImage(pil_mask)
        photo_refs.append(photo_mask)
        
        tk.Label(content, text="Tubule Mask", font=("Arial", 11, "bold"),
                bg=DARK_BG, fg=DARK_FG).grid(row=row_idx, column=0, pady=(5, 3), padx=10)
        img_label = tk.Label(content, image=photo_mask, bg=DARK_BG, relief="solid", bd=1)
        img_label.image = photo_mask
        img_label.grid(row=row_idx+1, column=0, padx=10, pady=(0, 15))
        
        # === CONTROL PANEL ===
        
        # Progress info
        progress_text = f"Progress: {idx+1} / {len(valid_files)}"
        tk.Label(controls, text=progress_text, font=("Arial", 12, "bold"),
                bg=DARK_BG, fg=DARK_FG).pack(pady=(0, 20))
        
        # Current state display
        tk.Label(controls, text="Current State:", font=("Arial", 11),
                bg=DARK_BG, fg=DARK_FG).pack(pady=(0, 5))
        tk.Label(controls, text=current_state, font=("Arial", 14, "bold"),
                bg=DARK_FRAME, fg=ACCENT_COLOR, relief="solid", bd=1,
                padx=10, pady=8).pack(fill="x", pady=(0, 20))
        
        # Accept/Reject buttons
        tk.Label(controls, text="Is this label correct?", font=("Arial", 11, "bold"),
                bg=DARK_BG, fg=DARK_FG).pack(pady=(0, 10))
        
        button_frame = tk.Frame(controls, bg=DARK_BG)
        button_frame.pack(fill="x", pady=(0, 20))
        
        # State selection frame (initially hidden)
        state_select_frame = tk.Frame(controls, bg=DARK_BG)
        
        def on_accept():
            user_response['accepted'] = True
            user_response['new_state'] = current_state
            root.quit()
        
        def on_reject():
            user_response['accepted'] = False
            # Show state selection
            tk.Label(state_select_frame, text="Select correct State:",
                    font=("Arial", 11, "bold"), bg=DARK_BG, fg=DARK_FG).pack(pady=(0, 10))
            
            # Create buttons for each unique state
            for state in unique_states:
                def make_state_callback(s):
                    return lambda: on_state_selected(s)
                
                btn = tk.Button(state_select_frame, text=state,
                               command=make_state_callback(state),
                               bg=BUTTON_BG, fg=BUTTON_FG,
                               font=("Arial", 10), cursor="hand2",
                               activebackground=BUTTON_HOVER,
                               bd=0, relief="flat")
                btn.pack(fill="x", pady=2)
            
            state_select_frame.pack(fill="x", pady=(10, 0))
        
        def on_state_selected(new_state):
            user_response['new_state'] = new_state
            root.quit()
        
        # Accept button
        btn_accept = tk.Button(button_frame, text="✓ Accept", command=on_accept,
                              bg=ACCEPT_COLOR, fg="white", font=("Arial", 12, "bold"),
                              cursor="hand2", activebackground="#45a049",
                              bd=0, relief="flat", height=2)
        btn_accept.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Reject button
        btn_reject = tk.Button(button_frame, text="✗ Reject", command=on_reject,
                              bg=REJECT_COLOR, fg="white", font=("Arial", 12, "bold"),
                              cursor="hand2", activebackground="#da190b",
                              bd=0, relief="flat", height=2)
        btn_reject.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # Skip button
        def on_skip():
            user_response['accepted'] = None
            user_response['new_state'] = None
            root.quit()
        
        tk.Button(controls, text="Skip", command=on_skip,
                 bg=BUTTON_BG, fg=BUTTON_FG, font=("Arial", 11),
                 cursor="hand2", activebackground=BUTTON_HOVER,
                 bd=0, relief="flat", height=2).pack(fill="x", pady=(0, 10))
        
        # Exit button
        def on_exit():
            global terminate
            terminate = True
            root.quit()
        
        tk.Button(controls, text="Exit & Save", command=on_exit,
                 bg=BUTTON_BG, fg=BUTTON_FG, font=("Arial", 11),
                 cursor="hand2", activebackground=BUTTON_HOVER,
                 bd=0, relief="flat", height=2).pack(fill="x")
        
        # Run GUI
        root.mainloop()
        
        # Process response
        if user_response['accepted'] is None:
            # Skipped
            print(f"  → Skipped")
            with open(skipped_txt, 'a') as f:
                f.write(f'{filename}\n')
        else:
            # Save result
            result = {
                'filename': filename,
                'sample_area': file_info.get('sample_area', ''),
                'tubule_id': file_info.get('tubule_id', ''),
                'classification': file_info.get('classification', ''),
                'original_state': current_state,
                'reviewed_state': user_response['new_state'],
                'accepted': user_response['accepted'],
                'centroid_x': file_info.get('centroid_x', ''),
                'centroid_y': file_info.get('centroid_y', '')
            }
            
            output_df = pd.concat([output_df, pd.DataFrame([result])], ignore_index=True)
            output_df.to_csv(output_csv, index=False)
            
            with open(reviewed_txt, 'a') as f:
                f.write(f'{filename}\n')
            
            status = "Accepted" if user_response['accepted'] else f"Rejected → {user_response['new_state']}"
            print(f"  → {status}")
        
        # Clean up
        try:
            root.destroy()
        except:
            pass
        
        if terminate:
            break
    
    # Final summary
    print("\n" + "="*70)
    print("REVIEW SESSION COMPLETE")
    print("="*70)
    print(f"Total reviewed: {len(output_df)}")
    if len(output_df) > 0:
        accepted = output_df['accepted'].sum()
        rejected = len(output_df) - accepted
        print(f"  Accepted: {accepted}")
        print(f"  Rejected & Corrected: {rejected}")
        print(f"\nResults saved to: {output_csv}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TUBULE PATCH LABEL REVIEW GUI")
    print("="*70)
    print(f"Configuration:")
    print(f"  Patch directory: {CONFIG['patch_dir']}")
    print(f"  Metadata CSV: {CONFIG['metadata_csv']}")
    print(f"  Output CSV: {CONFIG['output_csv']}")
    print(f"  CODEX channels: {CONFIG['codex_channels']}")
    print("="*70 + "\n")
    
    main()