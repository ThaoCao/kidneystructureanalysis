#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:30:20 2025

@author: thaocao
"""
import os
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import resize
import skimage
import matplotlib.pyplot as plt


rootdir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/'
datasets = ['Renal_Allograft']
tissue_mask = 'tissue_composite_masks'
norm_comp = 'Normalized_composites'
structure_overlay = 'structure_markers_overlay'
save_output = 'structure_markers_summed_ds_16bit_cleaned3'
marker_tubules_list = ['MUC-1', 'Claudin1', 'CD138', 'CD10_']

# Specify the sample and area to process
TARGET_SAMPLE_AREAS = ['011223S1_Area1','112222S2_Area3','112222S4_Area2','121522S1_Area1','121522S1_Area2','121522S2_Area1']

def process_structure_markers_overlay(dataset):
    rdir_struct = os.path.join(rootdir, dataset, tissue_mask)
    rdir_norm_comp = os.path.join(rootdir, dataset, norm_comp)
    sdir = os.path.join(rootdir, dataset, structure_overlay)

    if not os.path.exists(sdir):
        os.makedirs(sdir)

    sims = [f for f in os.listdir(rdir_struct) if os.path.isfile(os.path.join(rdir_struct, f))]
    sims.sort()

    for sim in sims:
        parts = sim.split('_')
        sample = parts[-2]
        area = parts[-1].split('.')[0]
        sample_area = f"{sample}_{area}"
        print('Processing ', sample_area)

        if sample_area not in TARGET_SAMPLE_AREAS:
            print(f"Skipping {sample}_{area} as it is not in the target list.")
            continue

        output_filename = f'{sample}_{area}_structure_markers_overlay.tif'
        output_path = os.path.join(sdir, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping {output_filename} as it already exists.")
            continue

        rdir_markers = os.path.join(rdir_norm_comp, sample, area)

        if not os.path.exists(rdir_markers):
            print(f"Skipping {sample}_{area} as the marker directory does not exist.")
            continue

        try:
            simg = imread(os.path.join(rdir_struct, sim))
        except FileNotFoundError:
            print(f"File not found: {os.path.join(rdir_struct, sim)}. Skipping.")
            continue

        markers_all = os.listdir(rdir_markers)
        markers_current = [x for x in markers_all if x.split('_')[0] in marker_tubules_list]

        marker_array = np.empty((len(markers_current), *simg.shape), dtype=np.uint16)

        for idx, marker in enumerate(markers_current):
            marker_path = os.path.join(rdir_markers, marker)
            if not os.path.exists(marker_path):
                print(f"Marker file not found: {marker_path}. Skipping.")
                continue
            this_marker = imread(marker_path)
            marker_array[idx] = this_marker

        imstack = np.concatenate([simg[np.newaxis], marker_array], axis=0)

        imwrite(output_path, imstack.astype(np.uint16))
        print(f"Saved {output_filename}")

    print("Processing complete for structure_markers_overlay.")


def process_structure_markers_summed(dataset):
    rdir_struct2 = os.path.join(rootdir, dataset, structure_overlay)
    rdir_tissue2 = os.path.join(rootdir, dataset, tissue_mask)
    sdir2 = os.path.join(rootdir, dataset, save_output)

    if not os.path.exists(sdir2):
        os.makedirs(sdir2)

    for sim in os.listdir(rdir_struct2):
        sample, area = sim.split('_')[:2]
        sample_area = f"{sample}_{area}"

        if sample_area not in TARGET_SAMPLE_AREAS:
            print(f"Skipping {sample}_{area} as it is not in the target list.")
            continue

        output_filename = f'{sample}_{area}_structure_overlay_ds10_16bit_cleaned.tif'
        output_path2 = os.path.join(sdir2, output_filename)

        if os.path.exists(output_path2):
            print(f"Skipping {output_filename} as it already exists.")
            continue

        try:
            simg = imread(os.path.join(rdir_struct2, sim))
        except FileNotFoundError:
            print(f"File not found: {os.path.join(rdir_struct2, sim)}. Skipping.")
            continue

        last_4_channels = simg[1:, :, :]
        sum_of_channels = np.sum(last_4_channels, axis=0)
        norm_sum = sum_of_channels / (65535 * 3)  # edited this to reflect the sum
        norm_sum_2 = 65535 * norm_sum / np.max(norm_sum)
        r, c = np.shape(norm_sum_2)
        ds_img_norm_cleaned = resize(norm_sum_2, (r // 10, c // 10), order=1, preserve_range=True)
        imwrite(output_path2, ds_img_norm_cleaned.astype(np.uint16))
        print(f"Saved {output_filename}")

    print(f"Processing complete for dataset {dataset}.")


def process_structure_markers_summed_plot(dataset):
    rdir_struct2 = os.path.join(rootdir, dataset, structure_overlay)
    rdir_tissue2 = os.path.join(rootdir, dataset, tissue_mask)
    sdir2 = os.path.join(rootdir, dataset, save_output)

    if not os.path.exists(sdir2):
        os.makedirs(sdir2)

    for sim in os.listdir(rdir_struct2):
        sample, area = sim.split('_')[:2]
        sample_area = f"{sample}_{area}"

        if sample_area not in TARGET_SAMPLE_AREAS:
            print(f"Skipping {sample}_{area} as it is not in the target list.")
            continue

        output_filename = f'{sample}_{area}_structure_overlay_ds10_16bit_cleaned.tif'
        output_path2 = os.path.join(sdir2, output_filename)

        if os.path.exists(output_path2):
            print(f"Skipping {output_filename} as it already exists.")
            continue

        try:
            simg = imread(os.path.join(rdir_struct2, sim))
        except FileNotFoundError:
            print(f"File not found: {os.path.join(rdir_struct2, sim)}. Skipping.")
            continue

        # Plot original image
        plt.figure(figsize=(10, 10))
        plt.imshow(simg[0], cmap='gray')
        plt.title('Original Image')
        plt.show()

        last_4_channels = simg[1:, :, :]
        sum_of_channels = np.sum(last_4_channels, axis=0)

        # Plot sum of channels
        plt.figure(figsize=(10, 10))
        plt.imshow(sum_of_channels, cmap='viridis')
        plt.title('Sum of Channels')
        plt.colorbar()
        plt.show()

        norm_sum = 255 * sum_of_channels / 65535
        norm_sum_2 = norm_sum / np.max(norm_sum)

        # Plot normalized sum
        plt.figure(figsize=(10, 10))
        plt.imshow(norm_sum_2, cmap='viridis')
        plt.title('Normalized Sum')
        plt.colorbar()
        plt.show()

        im2 = skimage.exposure.equalize_adapthist(norm_sum_2, kernel_size=None, clip_limit=0.01, nbins=256)

        # Plot after CLAHE
        plt.figure(figsize=(10, 10))
        plt.imshow(im2, cmap='viridis')
        plt.title('After CLAHE')
        plt.colorbar()
        plt.show()

        im3 = im2 * 255
        r, c = np.shape(im3)
        ds_img = resize(im3, (r // 10, c // 10), order=1, preserve_range=True)

        # Plot downsampled image
        plt.figure(figsize=(10, 10))
        plt.imshow(ds_img, cmap='viridis')
        plt.title('Downsampled Image')
        plt.colorbar()
        plt.show()

        p995 = np.percentile(ds_img, 99.5)
        p1 = np.percentile(ds_img, 1)
        ds_img_norm = (ds_img - p1) / (p995 - p1) * 255

        # Plot normalized downsampled image
        plt.figure(figsize=(10, 10))
        plt.imshow(ds_img_norm, cmap='viridis')
        plt.title('Normalized Downsampled Image')
        plt.colorbar()
        plt.show()

        tissue_mask_filename = f'DAPI_UV_1_{dataset}_{sample}_{area}.tif'
        tissue_mask_path = os.path.join(rdir_tissue2, tissue_mask_filename)

        if not os.path.exists(tissue_mask_path):
            print(f"Tissue mask file not found: {tissue_mask_path}. Skipping.")
            continue

        tissue_mask_img = imread(tissue_mask_path)
        r_ds, c_ds = np.shape(ds_img_norm)
        tissue_mask_resized = resize(tissue_mask_img, (r_ds, c_ds), order=3, preserve_range=True)
        tissue_mask_resized[tissue_mask_resized != 255] = 0

        # Plot tissue mask
        plt.figure(figsize=(10, 10))
        plt.imshow(tissue_mask_resized, cmap='gray')
        plt.title('Tissue Mask')
        plt.show()

        ds_img_norm_cleaned = ds_img_norm * tissue_mask_resized / 255

        # Plot final cleaned image
        plt.figure(figsize=(10, 10))
        plt.imshow(ds_img_norm_cleaned, cmap='viridis')
        plt.title('Final Cleaned Image')
        plt.colorbar()
        plt.show()

        imwrite(output_path2, ds_img_norm_cleaned.astype(np.uint8))
        print(f"Saved {output_filename}")

    print(f"Processing complete for dataset {dataset}.")


for dataset in datasets:
    process_structure_markers_overlay(dataset)
    process_structure_markers_summed(dataset)
    # process_structure_markers_summed_plot(dataset)  #Commented this out because it generates plots

print("All datasets processed.")
