#!/bin/bash

input_dir="/home/thaocao/samples_new/CD34/"
model_dir="/home/thaocao/omnipose_bv_mixed_training_2_CD31_sub_CD138/training/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_1_dim_2_training_2024_10_29_20_39_06.310439"

find "$input_dir" -type d -print0 | while IFS= read -r -d '' subfolder; do
    omnipose --dir "$subfolder" --use_gpu --pretrained_model "$model_dir" --nclasses 2 --nchan 1 --save_tif --in_folders --mask_threshold 0.2 --fast_mode --affinity_seg --omni --cluster --no_net_avg --diameter 22
done
