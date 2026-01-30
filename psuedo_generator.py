import os
import re
import numpy as np
from collections import defaultdict

RAW_DIR = "raw_npy"
FLAT_DIR = "flats"
OUT_DIR = "normalized_psuedo"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load flat-field images
# -----------------------------
flats = {}
for led in [1, 2, 3]:
    flat_path = os.path.join(FLAT_DIR, f"LED{led}_flat_ref.npy")
    flat = np.load(flat_path).astype(np.float32)
    flat[flat == 0] = 1.0  # avoid divide-by-zero
    flats[led] = flat

# -----------------------------
# Group raw files by (timestamp, color)
# -----------------------------
pattern = re.compile(
    r"(?P<ts>\d{8}-\d{6})_LED(?P<led>\d)_(?P<color>\w+)_raw\.npy"
)

groups = defaultdict(dict)

for fname in os.listdir(RAW_DIR):
    match = pattern.match(fname)
    if not match:
        continue

    ts = match.group("ts")
    led = int(match.group("led"))
    color = match.group("color")

    key = (ts, color)
    groups[key][led] = os.path.join(RAW_DIR, fname)

# -----------------------------
# Normalize, stack, save
# -----------------------------
for (ts, color), led_files in groups.items():
    if not all(led in led_files for led in [1, 2, 3]):
        print(f"⚠️ Skipping {ts}_{color} (missing LED)")
        continue

    norm_imgs = []

    for led in [1, 2, 3]:
        raw = np.load(led_files[led]).astype(np.float32)
        flat = flats[led]

        norm = raw / flat
        norm_imgs.append(norm)

    stacked = np.dstack(norm_imgs).astype(np.float32)

    out_name = f"{ts}_{color}_norm1_stack.npy"
    out_path = os.path.join(OUT_DIR, out_name)
    np.save(out_path, stacked)

    print(f"✅ Saved: {out_name} | shape={stacked.shape}")
