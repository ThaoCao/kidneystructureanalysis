import os
import shutil
from pathlib import Path

# ── Configure paths ────────────────────────────────────────────────────────────
REF_DIR = Path("/project/mclark/SCAMPI_datasets/Lupus_Nephritis_60ch/samples_new/tubules/")
SRC_DIR = Path("/project/mclark/SCAMPI_datasets/Lupus_Nephritis_60ch/tissue_composite_masks/")
DST_DIR = Path("/project/mclark/SCAMPI_datasets/Lupus_Nephritis_60ch/samples_new/tissue/")
# ──────────────────────────────────────────────────────────────────────────────

def copy_files_by_reference(ref_dir: Path, src_dir: Path, dst_dir: Path):
    # Collect reference filenames (stems only, no extension — adjust if needed)
    ref_names = [f.name for f in ref_dir.rglob("*") if f.is_file()]

    if not ref_names:
        print("No reference files found. Exiting.")
        return

    print(f"Found {len(ref_names)} reference filenames.")

    # Collect all source files
    src_files = [f for f in src_dir.rglob("*") if f.is_file()]
    print(f"Found {len(src_files)} source files to search through.\n")

    copied, skipped = 0, 0

    for ref_name in ref_names:
        # Find all source files whose name contains the reference name as substring
        matches = [f for f in src_files if ref_name in f.name]

        if not matches:
            print(f"  [NO MATCH] '{ref_name}'")
            skipped += 1
            continue

        for src_file in matches:
            # Preserve relative directory structure inside dst_dir
            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            if dst_file.exists():
                print(f"  [SKIP - exists] {dst_file.name}")
                continue

            shutil.copy2(src_file, dst_file)
            print(f"  [COPIED] {src_file.name}  →  {dst_file}")
            copied += 1

    print(f"\nDone. Copied: {copied} | Skipped (no match): {skipped}")


if __name__ == "__main__":
    copy_files_by_reference(REF_DIR, SRC_DIR, DST_DIR)
