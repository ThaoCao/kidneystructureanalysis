# Renal Damage Diagnosis (RDDx)

**RDDx is a high-throughput, modular computational pipeline for instance segmentation, phenotypic classification, and spatial analysis of kidney parenchymal structures in whole-slide multiplexed fluorescence images.**

---

## Overview of RDDx

RDDx pipeline preprocesses whole-slide fluorescence images as input and produces quantitative structural metrics that characterize kidney tissue in healthy and diseased states. The pipeline was developed and validated on a CODEX multiplexed fluorescence imaging dataset of 64 formalin-fixed paraffin-embedded (FFPE) kidney biopsies, encompassing 23 lupus nephritis (LuN), 33 renal allograft rejection (RAR; including antibody-mediated rejection [ABMR], T cell–mediated rejection [TCMR], and mixed rejection [MR]), and 8 kidney control (KC) biopsies.

RDDx operates in three sequential stages:

1. **Structural segmentation** — Three independent deep learning networks generate instance masks for glomeruli, tubules, and capillaries from multichannel fluorescence overlays.
2. **Phenotypic classification** — Segmented instances are classified by structural type and pathological state using fluorescence biomarker intensities and morphometric features.
3. **Spatial analysis** — Classified instances are analyzed for spatial clustering and co-occurrence patterns using Ripley's L statistics and conditional co-occurrence probability scores.

Across all three networks, RDDx achieves a mean F1 score of 0.9 at an intersection over union (IoU) threshold of 0.5 on independent, human-annotated validation sets. All networks maintain an F1 score above 0.8 across disease cohorts, and demonstrate zero-shot generalization to a held-out CODEX dataset using an alternative endothelial marker (CD34 substituting for CD31), achieving a mean F1 of 0.8 without fine-tuning.

---

## RDDx Modules

RDDx comprises three parallel deep learning networks, each trained independently to segment a distinct kidney structural compartment. The U-Net architecture was selected for its encoder-decoder design, which captures multi-scale contextual information suited to biomedical image segmentation. For tubules and capillaries — which exhibit greater morphological irregularity than glomeruli — an Omnipose variant of U-Net is used. Omnipose augments the standard U-Net backbone with flow field and distance field computations, enabling more accurate delineation of elongated and asymmetric structures.

### Glomeruli network

| Parameter | Value |
|---|---|
| Architecture | U-Net |
| Input markers | CD10, CD31, Claudin1, DAPI |
| Training tile size | 256 × 256 × 3 px |
| Epochs | 100 |
| Learning rate | 0.0001 |
| Training set | 78 WSIs, 1,124 annotated tiles |
| Test set | 56 WSIs, 289 instances |

### Tubule network

| Parameter | Value |
|---|---|
| Architecture | Omnipose U-Net |
| Input markers | CD10, MUC1, Claudin1, CD138 |
| Training tile size | 160 × 160 × 1 px |
| Epochs | 1,000 |
| Learning rate | 0.01 |
| Training set | 8 WSIs, 420 annotated tiles |
| Test set | 6 WSIs, 7,011 instances |
| Disease cohorts in training | LuN, ABMR, TCMR, MR |

### Capillary network

| Parameter | Value |
|---|---|
| Architecture | Omnipose U-Net |
| Input markers | CD31 (CD138 subtracted as background) |
| Training tile size | 320 × 320 × 1 px |
| Epochs | 1,000 |
| Learning rate | 0.01 |
| Training set | 8 WSIs, 36 annotated tiles |
| Test set | 7 WSIs, 11,442 instances |
| Disease cohorts in training | LuN, ABMR, TCMR, MR |

Training and validation datasets were curated by two expert annotators. Instance selections for training and testing were randomized and balanced to include equal representation from normal and diseased cohorts to mitigate disease-specific bias.

### Structural classification

After segmentation, each instance is classified as follows.

**Tubule type** — Proximal tubules are identified by CD10 expression at or above a 95th-percentile intensity threshold of 20; distal tubules are identified by MUC1 expression at or above the same threshold. For unclassified tubules, a secondary criterion using the CD10:MUC1 intensity ratio is applied: ratios below 0.95 are assigned as distal; ratios above 1.05 as proximal.

**Tubule state** — Four pathological states are defined using biomarker intensity thresholds applied to 95th-percentile pixel values:

| State | Criteria |
|---|---|
| Stressed | Claudin1 intensity ≥ 45, frequency > 10 pixels |
| Inflamed | CD45 intensity ≥ 52, frequency > 10 pixels |
| Stressed and inflamed | Meets both stressed and inflamed criteria |
| Atrophic | CD10⁻ MUC1⁻; size ≥ 69 pixels; circularity ≥ 25th percentile threshold (0.893) |
| Healthy | Does not meet any above criteria |

**Capillary state** — Inflamed capillaries are identified by CD45 intensity ≥ 50 with a minimum CD45 pixel frequency of 5. Proliferating capillaries are identified by co-localization of a Ki67 mask (generated using `scipy.ndimage` Gaussian blur) with the capillary instance mask. Double-positive (inflamed and proliferating) capillaries meet both criteria.

### Spatial analysis

**Ripley's L** — Spatial clustering of tubule states is quantified by computing Ripley's L as a function of radius (0–400 pixels, step 10 pixels). Centroid coordinates for each classified tubule are extracted and pairwise Euclidean distances are calculated. The Ripley's K statistic is computed at each radius by counting tubule pairs within that radius and normalizing by sampling area; Ripley's L is then derived as L(r) = √(K(r)/π). Higher Ripley's L scores indicate greater spatial clustering.

**Co-occurrence** — Spatial enrichment of inflamed capillaries near each tubule state is quantified as a normalized conditional probability score p(x|y)/p(x), where p(x|y) is the fraction of tubules of state y whose nearest neighbor within a given radius is an inflamed capillary x, and p(x) is the overall prevalence of inflamed capillaries in the biopsy.

---

## Applications

RDDx was developed for quantitative assessment of kidney tissue injury in the following contexts:

- **Disease characterization** — Quantification of tubular and capillary density and state composition across kidney disease cohorts, including LuN, ABMR, TCMR, and MR.
- **Spatial injury mapping** — Identification of clustered versus diffuse patterns of tubular and capillary inflammation within individual biopsies.
- **Clinical score correlation** — Spearman correlation of quantitative structural metrics (stressed, inflamed, atrophic tubule percentages; inflamed capillary density) with pathologist-assigned scores for tubulointerstitial inflammation (TI), interstitial fibrosis (TF), tubular atrophy (TA), and chronicity index (CI, CIG, CITI).
- **Generalization to new datasets** — Zero-shot inference on CODEX datasets with alternative antibody panels, without retraining.
- **Potential extension** — The modular design supports adaptation to other renal diseases (e.g., diabetic nephropathy) or to tissue types beyond the kidney with appropriate retraining.

Across the study dataset, RDDx identified and analyzed a total of 1,368 glomeruli, 911,352 tubules, and 443,260 capillaries.

---

## Getting Started with RDDx

### Requirements

- Python 3.8 or later
- PyTorch (GPU recommended; CUDA-compatible device with ≥ 8 GB VRAM if you want to train your own models)
- [Cellpose](https://github.com/MouseLand/cellpose) / [Omnipose](https://github.com/kevinjohncutler/omnipose) (for tubule and capillary networks)
- NumPy, SciPy, scikit-image, pandas
- All dependencies are listed in:
-  `linux_requirements.yml` for Linux and Windows systems
-  `mac_environment.yml` for Mac OS system

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ThaoCao/kidneystructureanalysis.git
cd kidneystructureanalysis
pip install -r requirements.txt
```

To use GPU acceleration, ensure that a CUDA-compatible version of PyTorch is installed matching your CUDA version. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

### Running pre-trained networks

Pre-trained model weights for all three networks (glomeruli, tubules, capillaries) are provided in the `models/` directory.

**Input format** — Each network accepts normalized, downsampled multichannel fluorescence overlays as input. Images should be preprocessed using the steps described in the `preprocessing/` folder: stitching and alignment (via [ASHLAR](https://github.com/labsyspharm/ashlar)), background subtraction using autofluorescence images, and spectral normalization (min-max normalization to the 99th percentile). 
Whole-slide images are downsampled by a factor of 10 before inference.

To run inference on a preprocessed fluorescence image:

```python
from rddx import segment_structures

# Provide paths to normalized multichannel input arrays (numpy format)
# For tubules: channels = [CD10, MUC1, Claudin1, CD138]
# For capillaries: channel = [CD31 - CD138]
# For glomeruli: channels = [CD10, CD31, Claudin1, DAPI]

masks = segment_structures(
    image_path="path/to/normalized_image.npy",
    structure="tubules",          # options: "tubules", "capillaries", "glomeruli"
    model_dir="models/",
    use_gpu=True
)
```

To run the full classification and spatial analysis pipeline after segmentation:

```python
from rddx import classify_tubules, classify_capillaries, compute_ripley_l, compute_cooccurrence

tubule_states = classify_tubules(masks, intensity_data)
capillary_states = classify_capillaries(masks, intensity_data)

ripley_scores = compute_ripley_l(tubule_states, radii=range(0, 400, 10))
cooccurrence_scores = compute_cooccurrence(capillary_states, tubule_states, radii=range(0, 400, 10))
```

Output is a DataFrame of per-instance classifications and per-biopsy summary statistics, including area percentages for each structural state.

### Training your own network

To train a new model on custom annotated data, place training tiles and ground-truth masks in the following directory structure:

```
data/
  train/
    images/        # multichannel input tiles (.tif or .npy)
    masks/         # instance segmentation ground truth
  test/
    images/
    masks/
```

Training and validation instances should be selected to include balanced representation from normal and diseased tissue to avoid disease-specific bias. Two or more expert annotators are recommended for ground-truth generation.

Launch training using the provided configuration:

```python
from rddx import train_model

train_model(
    structure="tubules",          # options: "tubules", "capillaries", "glomeruli"
    data_dir="data/",
    save_dir="models/custom/",
    n_epochs=1000,                # recommended: 1000 for tubules and capillaries, 100 for glomeruli
    learning_rate=0.01,           # recommended: 0.01 for tubules and capillaries, 0.0001 for glomeruli
    tile_size=160,                # 160 for tubules, 320 for capillaries, 256 for glomeruli
    use_gpu=True
)
```

The Omnipose framework is used by default for tubule and capillary training, given the morphological irregularity of these structures. The glomeruli network uses a standard U-Net. Refer to Table 4 of the accompanying paper for the full list of training hyperparameters used in the published models.

### Fine-tuning using refinement U-Nets

Pre-trained networks can be fine-tuned on new datasets with different antibody panels or imaging conditions. This is useful when canonical structural biomarkers are substituted (e.g., CD34 as an alternative endothelial marker in place of CD31 for capillaries), or when tissue quality or staining protocols differ from the original training set.

Fine-tuning requires a small set of manually annotated examples from the new dataset. As few as 1–2 whole-slide images (WSIs) with annotated tiles have been found sufficient for adaptation in zero-shot evaluation; performance on denser or more damaged tissue regions may benefit from additional annotations.

```python
from rddx import finetune_model

finetune_model(
    structure="capillaries",
    base_model_path="models/capillaries_pretrained.pth",
    data_dir="data/new_dataset/",
    save_dir="models/finetuned/",
    n_epochs=200,
    learning_rate=0.001,
    use_gpu=True
)
```

Note that tissue regions with severe damage, increased interstitial expansion, or sparse structural density may yield lower recall, as both network models and human annotators are less likely to identify structures in these areas.

### Validation

To evaluate network performance on a held-out annotated test set, use the provided evaluation script. Performance is reported as F1 score, precision, recall, and mean IoU across a range of IoU thresholds (0.2–0.85):

```python
from rddx import evaluate_model

results = evaluate_model(
    structure="tubules",
    model_path="models/tubules_pretrained.pth",
    test_data_dir="data/test/",
    iou_thresholds=[round(t, 2) for t in [x * 0.05 + 0.2 for x in range(14)]],
    use_gpu=True
)

results.to_csv("validation_results.csv", index=False)
```

The published pre-trained models achieve the following performance on independent test sets at IoU = 0.5:

| Network | Mean F1 (all cohorts) | F1 in KC | F1 in LuN | F1 in RAR |
|---|---|---|---|---|
| Glomeruli | ~0.9 | Highest | Lowest | Intermediate |
| Tubules | ~0.9 | Highest | Similar | Similar |
| Capillaries | ~0.9 | Highest | Similar | Similar |

All three networks maintain F1 > 0.8 across disease cohorts at IoU threshold of 0.5. Performance decreases at higher IoU thresholds (> 0.8) due to minor pixel-level boundary mismatches that do not affect object-level analysis outcomes. Failures are most common at tissue edges where incomplete structures occur.

---

## Citation

If you use RDDx in your research, please cite:

> Cao T, Torcasso MS, Ai J, Hara S, Andrade MS, Chang A, Casella G, Chong AS, Giger ML, Clark MR. *High-dimensional spatial proteomics and novel machine learning pipeline identifies disease specific renal damage states.* (2025).

---

## License

Please refer to `LICENSE` for terms of use.

---

## Acknowledgments

These studies were funded by the NIH Autoimmunity Centers of Excellence (AI082724), Department of Defense (LRI180083), Alliance for Lupus Research, Chan Zuckerberg Biohub, and NIH awards (S10-OD025081, S10-RR021039, and P30-CA14599). Imaging was performed at the University of Chicago Human Disease and Immune Discovery Core. Computational analyses were performed on the MEL computational server in the Radiomics and Machine Learning Facility at the University of Chicago.

The core segmentation models of RDDx are built upon the [Omnipose](https://omnipose.readthedocs.io/index.html) paper by Cutler et al. who introduced a high-precision morphology-independent solution for bacterial cell segmentation. Please visit [Omnipose paper](https://www.nature.com/articles/s41592-022-01639-4). RDDx uses code licensed under the MIT License from [Omnipose] (https://github.com/kevinjohncutler/omnipose).
---

## Contact

For questions, please open a GitHub issue or contact the corresponding author, Marcus Clark, at mclark@uchicago.edu.
