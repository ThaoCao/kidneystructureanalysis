#!/bin/bash

input_dir="/home/thaocao/HE_CODE_MODEL/HE_ds"
model_dir="/home/thaocao/omnipose_HE_mixed_training6/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_omnipose_HE_mixed_training6_2026_02_04_14_50_04.876145"
find "$input_dir" -type d -print0 | while IFS= read -r -d '' subfolder; do
    omnipose --dir "$subfolder" --use_gpu --pretrained_model "$model_dir" --nclasses 2 --nchan 3 --channel_axis 2 --save_tif --in_folders --mask_threshold 0.99 --fast_mode --affinity_seg --no_net_avg --diameter 30
done
