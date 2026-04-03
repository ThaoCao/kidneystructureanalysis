#!/bin/bash

input_dir="/home/thaocao/Normal_Kidney_60ch/glomeruli/"
model_dir="/home/thaocao/omnipose_glom_mixed_rgb_training1/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_omnipose_glom_mixed_rgb_training1_2025_02_17_14_56_23.861698"

find "$input_dir" -type d -print0 | while IFS= read -r -d '' subfolder; do
    omnipose --dir "$subfolder" --use_gpu --pretrained_model "$model_dir" --nclasses 2 --nchan 3 --save_tif --in_folders --mask_threshold 0.99 --fast_mode --affinity_seg --omni --cluster --no_net_avg --diameter 30
done
