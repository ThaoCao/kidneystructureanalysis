#!/bin/bash

input_dir="/nfs/kitbag/CellularImageAnalysis/SCAMPI_datasets/Lupus_Nephritis/HE_images_resized/"
model_dir="/home/thaocao/omnipose_mixed_training4/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_1_dim_2_omnipose_mixed_training4_2024_06_10_15_19_02.857846"
find "$input_dir" -type d -print0 | while IFS= read -r -d '' subfolder; do
    omnipose --dir "$subfolder" --use_gpu --pretrained_model "$model_dir" --nclasses 2 --nchan 1 --save_tif --in_folders --mask_threshold 0.99 --fast_mode --affinity_seg --omni --cluster --no_net_avg --diameter 30 --tyx 128,128
done
