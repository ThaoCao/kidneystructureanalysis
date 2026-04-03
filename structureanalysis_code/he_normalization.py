#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:38:12 2025

@author: thaocao
"""

import PIL.Image as Image
import os
from torchvision import transforms as transforms
import cv2
import numpy as np
from skimage import color
import tifffile
import glob

def quick_loop(image, image_avg, image_std, temp_avg, temp_std, isHed=False):
    image = (image - np.array(image_avg)) * (
        np.array(temp_std) / np.array(image_std)
    ) + np.array(temp_avg)
    if isHed:  # HED in range[0,1]
        pass
    else:  # LAB/HSV in range[0,255]
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)

def reinhard_cn(image_path, temp_path, save_path, isDebug=False, color_space=None):
    isHed = False
    
    # Read images using tifffile
    image = tifffile.imread(image_path)
    template = tifffile.imread(temp_path)
    
    # Transpose from (C, H, W) to (H, W, C) if needed
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
        if isDebug:
            print(f"  Transposed image from (3, H, W) to (H, W, 3)")
    
    if template.ndim == 3 and template.shape[0] == 3:
        template = np.transpose(template, (1, 2, 0))
        if isDebug:
            print(f"  Transposed template from (3, H, W) to (H, W, 3)")
    
    if isDebug:
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Template shape: {template.shape}, dtype: {template.dtype}")
        tifffile.imwrite("source.tif", image)
        tifffile.imwrite("template.tif", template)
    
    # Convert color spaces (now in RGB format with shape (H, W, 3))
    if color_space == "LAB":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # LAB range[0,255]
        template = cv2.cvtColor(template, cv2.COLOR_RGB2LAB)
    elif color_space == "HED":
        isHed = True
        image = color.rgb2hed(image)  # HED range[0,1]
        template = color.rgb2hed(template)
    elif color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        template = cv2.cvtColor(template, cv2.COLOR_RGB2HSV)
    elif color_space == "GRAY":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        tifffile.imwrite(save_path, image)
        return
    
    image_avg, image_std = getavgstd(image)
    template_avg, template_std = getavgstd(template)
    
    if isDebug:
        print("isDebug!!!")
        print("source_avg: ", image_avg)
        print("source_std: ", image_std)
        print("target_avg: ", template_avg)
        print("target_std: ", template_std)
    
    # Reinhard's Method to Stain Normalization
    image = quick_loop(
        image, image_avg, image_std, template_avg, template_std, isHed=isHed
    )
    
    # Convert back and save using tifffile
    if color_space == "LAB":
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        tifffile.imwrite(save_path, image)
    elif color_space == "HED":  # HED[0,1]->RGB[0,255]
        image = color.hed2rgb(image)
        imin = image.min()
        imax = image.max()
        image = (255 * (image - imin) / (imax - imin)).astype("uint8")
        tifffile.imwrite(save_path, image)
    elif color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        tifffile.imwrite(save_path, image)
    
    if isDebug:
        tifffile.imwrite("results.tif", image)

if __name__ == "__main__":
    # Define directories
    input_dir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Renal_Allograft/HE_images/'
    template_path = "/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Lupus_Nephritis/HE_images/LuN_012523S4_Area5.tif"  # Template image path
    output_dir = '/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Renal_Allograft/HE_normalized'
   
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all TIFF images from input directory
    # This supports .tif, .tiff, .TIF, .TIFF extensions
    tiff_patterns = [
        os.path.join(input_dir, "*.tif"),
        os.path.join(input_dir, "*.tiff"),
        os.path.join(input_dir, "*.TIF"),
        os.path.join(input_dir, "*.TIFF")
    ]
    
    img_path_list = []
    for pattern in tiff_patterns:
        img_path_list.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    img_path_list = sorted(list(set(img_path_list)))
    
    print(f"Found {len(img_path_list)} TIFF images in {input_dir}")
    print(f"Using template: {template_path}")
    print(f"Saving normalized images to: {output_dir}")
    print("-" * 50)
    
    # Process each image
    for idx, img_path in enumerate(img_path_list, 1):
        # Get filename from path
        filename = os.path.basename(img_path)
        
        # Create save path
        save_path = os.path.join(output_dir, filename)
        
        print(f"[{idx}/{len(img_path_list)}] Processing: {filename}")
        
        # Normalize and save
        try:
            img_colorNorm = reinhard_cn(
                img_path, template_path, save_path, isDebug=False, color_space="LAB"
            )
            print(f"  ✓ Saved to: {save_path}")
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
    
    print("-" * 50)
    print(f"Normalization complete! Processed {len(img_path_list)} images.")

