#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from skimage.measure import regionprops
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===================== CONFIG =====================
ROOT_DIR = "/project/mclark/SCAMPI_datasets/Duke_TCMR_60ch/"
TUBULE_MASK_DIR = os.path.join(ROOT_DIR, "tubules",'cleaned')

INPUT_CSV  = os.path.join(ROOT_DIR, "tubules_features/tubules_px_all_95th.csv")

OUTPUT_DIR = os.path.join(ROOT_DIR, "tubules/states")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ALL_SAMPLES_tubules_classified.csv")

OUTPUT_IMG_DIR_TYPES  = os.path.join(ROOT_DIR, "tubules/types/")
OUTPUT_IMG_DIR_STATES = os.path.join(ROOT_DIR, "tubules/states/")

# Ensure all output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR_TYPES, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR_STATES, exist_ok=True)

# NOTE: if you are I/O bound, 4–8 may outperform 16
NUM_WORKERS = 16

UPSAMPLE_FACTOR = 10
WRITE_UPSAMPLED_OVERLAYS = True

# ===================== Lupus thresholds =====================
THR = {
    "inside": {
        "p95": {"CD10": 20, "MUC1": 20, "Claudin1": 20, "CD45": 40},
        "frac": {"Claudin1": 0.0803, "CD45": 0.0777},  # pos_frac in 0–1
    }
}

SIZE_SHRINKAGE_MULTIPLIER = 0.7
LOW_SIGNAL_FLOOR = 15
CIRCULARITY_THRESHOLD = 0.6

# ---- Colors for overlay ----
TYPE_COLORS = {
    "Proximal": (153, 255, 0),
    "Distal": (0, 153, 255),
    "Unclassified": (102, 102, 153),
}
STATE_COLORS = {
    "Healthy": (184,225,134),
    "Stressed": (241,182,218),
    "Inflamed": (253,184,99),
    "Stressed and Inflamed": (202,0,32),
    "Atrophic": (150,100,100),
    "Unclassified": (128,128,128),
}

AREA_RE = re.compile(r"(Area\d+)", re.IGNORECASE)

# ===================== Mask index (speed) =====================
def build_mask_index(mask_dir: str):
    """
    Build {(sample, area): filepath} using your filename convention that contains AreaX.
    Example filename: Sample123_Area1.tif
    """
    idx = {}
    for fn in os.listdir(mask_dir):
        if not fn.lower().endswith(".tif"):
            continue
        base = os.path.splitext(fn)[0]
        m = AREA_RE.search(base)
        if not m:
            continue
        area = m.group(1)
        sample = base[:m.start()].rstrip("_")
        if sample:
            idx[(sample, area)] = os.path.join(mask_dir, fn)
    return idx

# ===================== Harmonization helpers =====================
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expected columns for Lupus reclassification:
      - needs Sample, Area, TubuleID(or ID)
      - uses *_95th and *_pos_frac from your old pipeline
      - if someone provides percent-based *_freq, convert to *_pos_frac
      - uses Size (high-res) from your old pipeline; fallback to size
    """
    df = df.copy()

    # ---- ID column ----
    if "TubuleID" not in df.columns:
        if "ID" in df.columns:
            df["TubuleID"] = df["ID"].astype(int)
        else:
            raise ValueError("Need TubuleID or ID column in input CSV.")

    # ---- required grouping columns ----
    for c in ["Sample", "Area"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in input CSV.")

    # ---- ensure p95 columns exist ----
    p95_map = {
        "CD10_95th": "CD10",
        "MUC1_95th": "MUC1",
        "Claudin1_95th": "Claudin1",
        "CD45_95th": "CD45",
    }
    for p95_col, raw_col in p95_map.items():
        if p95_col not in df.columns and raw_col in df.columns:
            df[p95_col] = df[raw_col]

    needed_p95 = ["CD10_95th", "MUC1_95th", "Claudin1_95th", "CD45_95th"]
    missing_p95 = [c for c in needed_p95 if c not in df.columns]
    if missing_p95:
        raise ValueError(f"Missing required p95 columns in CSV: {missing_p95}")

    # ---- ensure pos_frac columns exist (0–1) ----
    for m in ["Claudin1", "CD45"]:
        frac_col = f"{m}_pos_frac"
        freq_col = f"{m}_freq"
        if frac_col not in df.columns and freq_col in df.columns:
            df[frac_col] = df[freq_col].astype(float) / 100.0

    needed_frac = ["Claudin1_pos_frac", "CD45_pos_frac"]
    missing_frac = [c for c in needed_frac if c not in df.columns]
    if missing_frac:
        raise ValueError(f"Missing required pos_frac columns in CSV: {missing_frac}")

    # ---- size column ----
    if "Size" not in df.columns:
        if "size" in df.columns:
            df["Size"] = df["size"]
        else:
            raise ValueError("Need Size (or size) column in input CSV for atrophy gating.")

    # ---- circularity ----
    if "Circularity" not in df.columns:
        df["Circularity"] = 0.0

    return df

# ===================== Circularity (FAST: one pass) =====================
def circularity_map_one_pass(tubule_mask: np.ndarray) -> dict:
    """
    Compute circularity for ALL labeled IDs in ONE PASS.
    Circularity = 4πA / P^2, capped at 1.0
    """
    circ_map = {}
    for p in regionprops(tubule_mask):
        tid = int(p.label)
        area = float(p.area)
        perim = float(p.perimeter)
        if perim > 0:
            circ = (4.0 * np.pi * area) / (perim ** 2)
            circ_map[tid] = float(min(circ, 1.0))
        else:
            circ_map[tid] = 0.0
    return circ_map

# ===================== Lupus classification logic (vectorized per-area) =====================
def classify_lupus_area(df_area: pd.DataFrame) -> pd.DataFrame:
    df = df_area.copy()
    if df.empty:
        return df

    # ---- Type_Label (mutually exclusive +/- like your original) ----
    df["Type_Label"] = "Unclassified"
    cd10_hi = df["CD10_95th"] >= THR["inside"]["p95"]["CD10"]
    muc1_hi = df["MUC1_95th"] >= THR["inside"]["p95"]["MUC1"]

    prox = cd10_hi & (~muc1_hi)   # +/-
    dist = muc1_hi & (~cd10_hi)   # -/+
    df.loc[prox, "Type_Label"] = "Proximal"
    df.loc[dist, "Type_Label"] = "Distal"

    # ---- Pathology flags ----
    stressed = (df["Claudin1_95th"] >= THR["inside"]["p95"]["Claudin1"]) & \
               (df["Claudin1_pos_frac"] >= THR["inside"]["frac"]["Claudin1"])
    inflamed = (df["CD45_95th"] >= THR["inside"]["p95"]["CD45"]) & \
               (df["CD45_pos_frac"] >= THR["inside"]["frac"]["CD45"])

    df["Stressed"] = stressed.astype(np.uint8)
    df["Inflamed"] = inflamed.astype(np.uint8)

    # ---- Atrophy gate (your original) ----
    median_size = df["Size"].median()
    size_cutoff = float(median_size) * float(SIZE_SHRINKAGE_MULTIPLIER)

    is_unclassified = (df["Type_Label"] == "Unclassified")
    no_pathology = (df["Stressed"] == 0) & (df["Inflamed"] == 0)

    is_small = (df["Size"] < size_cutoff)
    is_deformed = (df["Circularity"] < CIRCULARITY_THRESHOLD)
    is_ghost = (df["CD10_95th"] < LOW_SIGNAL_FLOOR) & (df["MUC1_95th"] < LOW_SIGNAL_FLOOR)

    df["Atrophic"] = 0
    df.loc[is_unclassified & no_pathology & (is_small & (is_ghost | is_deformed)), "Atrophic"] = 1

    # ---- Healthy ----
    df["Healthy"] = 0
    df.loc[(df["Stressed"] == 0) & (df["Inflamed"] == 0) & (df["Atrophic"] == 0), "Healthy"] = 1

    # ---- State_Label (FAST: vectorized) ----
    conds = [
        df["Atrophic"] == 1,
        (df["Stressed"] == 1) & (df["Inflamed"] == 1),
        df["Stressed"] == 1,
        df["Inflamed"] == 1,
        df["Healthy"] == 1,
    ]
    choices = ["Atrophic", "Stressed and Inflamed", "Stressed", "Inflamed", "Healthy"]
    df["State_Label"] = np.select(conds, choices, default="Unclassified")

    return df

# ===================== Overlay writer =====================
def write_overlays_from_mask(tubule_mask: np.ndarray, df_area: pd.DataFrame,
                             out_types_path: str, out_states_path: str,
                             upsample_factor: int = 1) -> None:
    """
    Paint overlay images in mask space using TubuleID -> (Type_Label, State_Label),
    then optionally upsample the RGB overlays by nearest-neighbor repeat.
    """
    H, W = tubule_mask.shape
    type_img  = np.zeros((H, W, 3), dtype=np.uint8)
    state_img = np.zeros((H, W, 3), dtype=np.uint8)

    lut = df_area.set_index("TubuleID")[["Type_Label", "State_Label"]].to_dict("index")

    ids = np.unique(tubule_mask)
    ids = ids[ids != 0]

    for tid in ids:
        info = lut.get(int(tid), {"Type_Label": "Unclassified", "State_Label": "Unclassified"})
        tcol = TYPE_COLORS.get(info["Type_Label"], (128, 128, 128))
        scol = STATE_COLORS.get(info["State_Label"], (128, 128, 128))

        m = (tubule_mask == tid)
        type_img[m]  = tcol
        state_img[m] = scol

    if upsample_factor and upsample_factor > 1:
        f = int(upsample_factor)
        type_img  = np.repeat(np.repeat(type_img,  f, axis=0), f, axis=1)
        state_img = np.repeat(np.repeat(state_img, f, axis=0), f, axis=1)

    imwrite(out_types_path, type_img)
    imwrite(out_states_path, state_img)

# ===================== Worker =====================
def process_one_area(sample: str, area: str, df_area: pd.DataFrame, mask_file: str) -> pd.DataFrame:
    """
    Load instance mask, recompute circularity (one pass),
    apply Lupus classification, and write overlays.
    """
    try:
        out = df_area.copy()

        if mask_file is None or (not os.path.exists(mask_file)):
            out = classify_lupus_area(out)
            out["_mask_note"] = "mask_not_found"
            return out

        tubule_mask = imread(mask_file)

        # FAST circularity lookup
        circ_map = circularity_map_one_pass(tubule_mask)
        out["Circularity"] = out["TubuleID"].astype(int).map(circ_map).fillna(0.0).astype(float)

        # classify
        out = classify_lupus_area(out)
        out["_mask_note"] = os.path.basename(mask_file)

        # overlays
        os.makedirs(OUTPUT_IMG_DIR_TYPES, exist_ok=True)
        os.makedirs(OUTPUT_IMG_DIR_STATES, exist_ok=True)
        base_name = f"{sample}_{area}"
        out_types  = os.path.join(OUTPUT_IMG_DIR_TYPES,  f"{base_name}_Types.tif")
        out_states = os.path.join(OUTPUT_IMG_DIR_STATES, f"{base_name}_States.tif")

        f = UPSAMPLE_FACTOR if WRITE_UPSAMPLED_OVERLAYS else 1
        write_overlays_from_mask(tubule_mask, out, out_types, out_states, upsample_factor=f)

        return out
    
    except Exception as e:
        print(f"ERROR processing {sample}_{area}: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty result to allow other processes to continue
        return pd.DataFrame()

# ===================== Main =====================
def main():
    print(f"Reading: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Initial rows: {len(df)}")

    df = df.drop_duplicates()
    print(f"After drop_duplicates: {len(df)}")

    df = ensure_columns(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR_TYPES, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR_STATES, exist_ok=True)

    # Build mask index once (speed)
    print("Indexing masks...")
    mask_index = build_mask_index(TUBULE_MASK_DIR)
    print(f"Indexed {len(mask_index)} masks from: {TUBULE_MASK_DIR}")

    groups = list(df.groupby(["Sample", "Area"], dropna=False))
    print(f"Found {len(groups)} (Sample, Area) groups. Using {NUM_WORKERS} workers.")

    results = []
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {}
        for (sample, area), df_area in groups:
            sample = str(sample)
            area = str(area)
            mask_file = mask_index.get((sample, area), None)
            fut = ex.submit(process_one_area, sample, area, df_area, mask_file)
            futures[fut] = (sample, area)

        for fut in as_completed(futures):
            sample, area = futures[fut]
            try:
                result = fut.result()
                if len(result) > 0:
                    results.append(result)
                    print(f"✓ Completed {sample}_{area}")
                else:
                    failed_count += 1
                    print(f"✗ Failed {sample}_{area} (returned empty)")
            except Exception as e:
                failed_count += 1
                print(f"✗ Exception for {sample}_{area}: {type(e).__name__}: {str(e)}")

    if not results:
        raise RuntimeError("All processing tasks failed. Check error messages above.")
    
    if failed_count > 0:
        print(f"\n⚠ Warning: {failed_count} groups failed to process")

    out = pd.concat(results, ignore_index=True)

    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY (LUPUS LOGIC)")
    print("="*60)
    print("\nState Distribution:")
    print(out["State_Label"].value_counts(dropna=False))
    print("\nType Distribution:")
    print(out["Type_Label"].value_counts(dropna=False))

    out.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*60)
    print(f"✓ Saved: {OUTPUT_CSV}")
    print(f"✓ Types overlays:  {OUTPUT_IMG_DIR_TYPES}")
    print(f"✓ States overlays: {OUTPUT_IMG_DIR_STATES}")
    print(f"✓ Final rows: {len(out)}")
    print("="*60)

if __name__ == "__main__":
    main()
