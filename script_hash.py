# run_hashing_test.py
import os, glob, pathlib
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from phaser.hashing import ComputeHashes, PHash, WaveHash, PdqHash
from phaser.transformers import Border, Crop, Flip, Rescale, Rotate, Watermark

ALGOS_dict = {"phash": PHash(hash_size=8), "wave": WaveHash(), "pdq": PdqHash()}
TRANS_list = [
    Border(border_colour=(255, 255, 255), border_width=30),
    Crop(cropbox_factors=[0.05, 0.05, 0.05, 0.05]),
    Flip(direction="Horizontal"),
    Rescale(fixed_dimensions=(96, 96), thumbnail_aspect=True),
    Rotate(degrees_counter_clockwise=5),
    Watermark(),
]
METR_dict  = {"Hamming": "hamming", "Cosine": "cosine"}

ROOT_DIR   = r"D:\NT334.Q21.ANTT\PHASER-main\PHASER-main"
OUTPUT_DIR = r"D:\NT334.Q21.ANTT\PHASER-main\PHASER-main\hashes"
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for file_id in range(1):  # ← test images0
    base_name  = f"images{file_id}"
    images_dir = os.path.join(ROOT_DIR, base_name, "images")

    if not os.path.exists(images_dir):
        print(f"[SKIP] {images_dir} không tồn tại")
        continue

    folder_ids = list(range(10))
    pbar = tqdm(folder_ids, desc=f"{base_name}", unit="folder")

    for folder_id in pbar:
        folder_path = os.path.join(images_dir, str(folder_id))
        out_name    = f"Hashes_{base_name}_{folder_id}.df.bz2"
        out_path    = os.path.join(OUTPUT_DIR, out_name)

        pbar.set_postfix({"folder": folder_id, "status": "checking"})

        if os.path.exists(out_path):
            pbar.set_postfix({"folder": folder_id, "status": "SKIP"})
            continue
        if not os.path.exists(folder_path):
            pbar.set_postfix({"folder": folder_id, "status": "NOT FOUND"})
            continue

        image_files = sorted(
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.png")) +
            glob.glob(os.path.join(folder_path, "*.jpeg"))
        )
        if not image_files:
            pbar.set_postfix({"folder": folder_id, "status": "NO IMAGES"})
            continue

        pbar.set_postfix({"folder": folder_id, "imgs": len(image_files), "status": "hashing..."})

        try:
            ch   = ComputeHashes(ALGOS_dict, TRANS_list, n_jobs=-1, progress_bar=True)
            df_h = ch.fit(image_files)
            print(f"\n  Transforms: {sorted(df_h['transformation'].unique())}")
            df_h["filename"] = df_h["filename"].apply(
                lambda x: f"{base_name}/{folder_id}/{x}"
            )
            dump(df_h, out_path, compress=9)
            pbar.set_postfix({"folder": folder_id, "rows": len(df_h), "status": "DONE"})
        except Exception as e:
            pbar.set_postfix({"folder": folder_id, "status": f"ERROR: {e}"})
            continue

print("\n✅ Phase 1 test xong!")