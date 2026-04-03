# Kidney Structural Analysis Pipeline
Goal: A deep learning pipeline that trains on multiplexed high-dimensional CODEX data to identify kidney structures and their states solely based on H&E input. 
Steps:
1. Generate rgb patch (input_rgb_patch_generate.py)
2. QC some random pairs of CODEX and H&E patches (input_rgb_patch_mask_qc.py)
3. Contrastive learning model training (model_constrastive.py)
4. Deploy the best trained model on WSI H&E (model_contrastive_deploy_WSI.py)
   
