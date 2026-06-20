# experiment1_convergence.py
# Reproduces Section 4.1 of the PHASER paper:
# "How many images?: Convergence of measures at various dataset sizes"
#
# Samples 1000 / 10,000 / 100,000 / 250,000 images from the full hash DataFrame,
# calculates inter- and intra-distances, computes EER Decision Thresholds,
# repeats 250 times per sample size, and produces boxen-plots (Fig. 4 style).
#
# Usage examples:
#   python experiment1_convergence.py
#   python experiment1_convergence.py --sizes 1000 10000 --iters 50 --seed 123
#   python experiment1_convergence.py --sizes 1000 --algos pdq --transforms Rescale Flip
#   python experiment1_convergence.py --seed 0          # different seed, still reproducible
#   python experiment1_convergence.py --list-options    # print available algos & transforms then exit

import os
import pathlib
import argparse
import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm
from scipy.spatial.distance import hamming
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Defaults (paper §4.1 values) ────────────────────────────────────────────
DEFAULT_SIZES      = [1_000, 10_000, 100_000, 250_000]
DEFAULT_ITERATIONS = 250
DEFAULT_SEED       = 42
DEFAULT_ALGOS      = ["pdq", "wave"]
DEFAULT_TRANSFORMS = ["Rescale", "Watermark", "Flip"]

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="PHASER Experiment 1 – EER threshold convergence across dataset sizes",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--sizes", nargs="+", type=int,
    default=DEFAULT_SIZES,
    metavar="N",
    help=f"Sample sizes to test (default: {DEFAULT_SIZES})\n"
         "Example: --sizes 1000 10000 100000",
)
parser.add_argument(
    "--iters", type=int,
    default=DEFAULT_ITERATIONS,
    metavar="N",
    help=f"Number of iterations per sample size (default: {DEFAULT_ITERATIONS})",
)
parser.add_argument(
    "--seed", type=int,
    default=DEFAULT_SEED,
    metavar="N",
    help=f"Random seed for reproducibility (default: {DEFAULT_SEED})\n"
         "Use the same seed to get identical results across runs.",
)
parser.add_argument(
    "--algos", nargs="+",
    default=DEFAULT_ALGOS,
    metavar="ALGO",
    help=f"Hash algorithms to include (default: {DEFAULT_ALGOS})\n"
         "Example: --algos pdq wave phash\n"
         "Run --list-options to see what is available in your data.",
)
parser.add_argument(
    "--transforms", nargs="+",
    default=DEFAULT_TRANSFORMS,
    metavar="TRANSFORM",
    help=f"Transforms to include (default: {DEFAULT_TRANSFORMS})\n"
         "Example: --transforms Rescale Flip Border\n"
         "Run --list-options to see what is available in your data.",
)
parser.add_argument(
    "--list-options", action="store_true",
    help="Load the data, print available algos and transforms, then exit.",
)

args = parser.parse_args()

SAMPLE_SIZES  = sorted(set(args.sizes))
N_ITERATIONS  = args.iters
RANDOM_SEED   = args.seed
TARGET_ALGOS  = args.algos
TARGET_TRANS  = args.transforms

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_OUTPUT   = r".\output"
EXP1_OUT      = os.path.join(BASE_OUTPUT, "experiment1_convergence")
pathlib.Path(EXP1_OUT).mkdir(parents=True, exist_ok=True)

HASHES_PATH   = os.path.join(BASE_OUTPUT, "Hashes.df.bz2")
ENCODERS_PATH = os.path.join(BASE_OUTPUT, "LabelEncoders.bz2")

DISTANCE_METRIC = "Hamming"   # paper uses only Hamming in §4.1

print("=" * 60)
print("PHASER – Experiment 1: EER Convergence")
print("=" * 60)
print(f"  Sample sizes : {SAMPLE_SIZES}")
print(f"  Iterations   : {N_ITERATIONS}")
print(f"  Random seed  : {RANDOM_SEED}  ← use same value to reproduce")
print(f"  Algorithms   : {TARGET_ALGOS}")
print(f"  Transforms   : {TARGET_TRANS}")
print(f"  Output dir   : {EXP1_OUT}")
print("=" * 60)

# ── --list-options: read only LabelEncoders (fast, skips the large hash file) ─
if args.list_options:
    print("Loading label encoders …")
    le = load(ENCODERS_PATH)
    print("\n── Available options for your data ──────────────────────────")
    print(f"  --algos      : {sorted(le['a'].classes_.tolist())}")
    print(f"  --transforms : {sorted(le['t'].classes_.tolist())}")
    print("─────────────────────────────────────────────────────────────")
    raise SystemExit(0)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading hashes …")
df_all = load(HASHES_PATH)
print(f"  Total rows : {len(df_all):,}")
print(f"  Columns    : {list(df_all.columns)}")

print("Loading label encoders ...")
le = load(ENCODERS_PATH)

# ── Detect format: wide (phash/wave/pdq as columns) vs long (algo column) ────
KNOWN_ALGO_COLS = {"phash", "wave", "pdq", "wavehash"}
algo_cols_found = [c for c in df_all.columns if c.lower() in KNOWN_ALGO_COLS]

if algo_cols_found:
    # Wide format: each algo is its own column -> melt into long format
    print(f"  Detected WIDE format. Algo columns: {algo_cols_found}")
    print("  Melting to long format ...")
    df_all = pd.melt(
        df_all,
        id_vars=["filename", "transformation"],
        value_vars=algo_cols_found,
        var_name="algo_short",
        value_name="hash",
    )
    df_all["algo_short"] = df_all["algo_short"].str.lower()
    df_all["trans_str"]  = le["t"].inverse_transform(df_all["transformation"].astype(int))
else:
    # Long format: single algo column encoded as integer
    print("  Detected LONG format.")
    df_all = df_all.copy()
    df_all["algo_short"] = le["a"].inverse_transform(df_all["algo"].astype(int))
    df_all["trans_str"]  = le["t"].inverse_transform(df_all["transformation"].astype(int))

# Normalise transform names to a clean single word
# Handles formats like:
#   "orig"                                           -> "orig"
#   "Flip"                                           -> "Flip"
#   "phaser.transformers._transforms.Flip"           -> "Flip"
#   "Rescale(fixed_dimensions=(96, 96), ...)"        -> "Rescale"
#   "Border(border_colour=(255, 255, 255), ...)"     -> "Border"
import re as _re
def short_name(name: str) -> str:
    s = str(name).strip()
    # take last dotted segment
    s = s.split(".")[-1]
    # take only the leading word (stops at "(", "_", " ", digit)
    m = _re.match(r"([A-Za-z]+)", s)
    return m.group(1) if m else s

df_all["trans_short"] = df_all["trans_str"].apply(short_name)

print(f"  Unique algos      : {sorted(df_all['algo_short'].unique())}")
print(f"  Short transforms  : {sorted(df_all['trans_short'].unique())}")
print(f"  Total rows (long) : {len(df_all):,}")

# ── Helper: normalise any hash representation → float32 bit array ────────────
def hash_to_bits(h):
    """
    Handles all formats PHASER may store hashes in:
      - numpy bool array  (True/False per bit)  ← your actual format
      - numpy int / Python int                  (e.g. 12345)
      - binary string                           (e.g. "0110101...")
      - hex string                              (e.g. "f3a0...")
    """
    # numpy bool array (True/False) — your actual format
    if isinstance(h, np.ndarray):
        return h.astype(np.float32)
    # plain Python bool list / sequence
    if isinstance(h, (list, tuple)) and len(h) > 0 and isinstance(h[0], (bool, np.bool_)):
        return np.array(h, dtype=np.float32)
    # integer
    if isinstance(h, (int, np.integer)):
        return np.array(list(bin(int(h))[2:]), dtype=np.float32)
    # string forms
    s = str(h).strip()
    if set(s) <= {"0", "1"}:           # binary string "01101..."
        return np.array(list(s), dtype=np.float32)
    try:                               # hex string "f3a0..."
        val = int(s, 16)
        return np.array(list(bin(val)[2:]), dtype=np.float32)
    except ValueError:
        raise ValueError(f"Cannot convert hash to bits: {repr(h)[:80]}")

# ── Helper: compute normalised Hamming distance between two hashes ─────────
def hamming_dist(h1, h2):
    b1, b2 = hash_to_bits(h1), hash_to_bits(h2)
    # pad shorter to same length
    diff = len(b1) - len(b2)
    if diff > 0:
        b2 = np.pad(b2, (0, diff))
    elif diff < 0:
        b1 = np.pad(b1, (0, -diff))
    return hamming(b1, b2)   # scipy hamming = normalised (0..1)

# ── Helper: compute EER threshold from labels and scores ─────────────────────
def compute_eer_threshold(y_true, scores):
    """
    y_true : 1 = intra (match), 0 = inter (non-match)
    scores : similarity (1 - hamming_distance), higher = more similar
    Returns the similarity threshold where FPR ≈ FNR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    # find index where |FPR - FNR| is minimised
    idx = np.argmin(np.abs(fpr - fnr))
    return float(thresholds[idx])

# ── Helper: get hash value safely from a df cell ─────────────────────────────
def get_hash(df_indexed, fname):
    """Get hash for a filename from an indexed DataFrame, handling duplicate indices."""
    val = df_indexed.loc[fname, "hash"]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return val

# ── Helper: sample n images and build inter/intra distance observations ───────
def build_observations(df_orig, df_trans, n_images, rng):
    """
    df_orig  : all 'orig' rows for a single algo  (indexed by filename int)
    df_trans : all rows for one specific transform for the same algo
               (indexed by filename int)
    n_images : number of unique images to sample
    Returns (labels, similarities) or (None, None) if not enough data.

    Intra  (label=1): similarity between orig hash and transform hash
                      for the SAME image  → should be HIGH (similar)
    Inter  (label=0): similarity between orig hash of image A and
                      orig hash of image B  → should be LOW (different)
    """
    # Images that have BOTH an orig hash and a transform hash
    common_imgs = np.intersect1d(df_orig.index.unique(), df_trans.index.unique())
    if len(common_imgs) < n_images:
        return None, None

    sampled = rng.choice(common_imgs, size=n_images, replace=False)

    # ── Intra: orig vs transform for same image ────────────────────────────
    intra_sims = []
    for fname in sampled:
        h_o = get_hash(df_orig,  fname)
        h_t = get_hash(df_trans, fname)
        intra_sims.append(1.0 - hamming_dist(h_o, h_t))

    n_intra = len(intra_sims)
    if n_intra == 0:
        return None, None

    # ── Inter: orig vs orig for DIFFERENT images ───────────────────────────
    # Build a lookup array of orig hashes for the sampled images
    orig_hashes = np.array([get_hash(df_orig, f) for f in sampled], dtype=object)
    n = len(sampled)

    inter_sims = []
    attempts   = 0
    max_att    = n_intra * 20
    while len(inter_sims) < n_intra and attempts < max_att:
        i, j = rng.choice(n, size=2, replace=False)
        inter_sims.append(1.0 - hamming_dist(orig_hashes[i], orig_hashes[j]))
        attempts += 1

    if len(inter_sims) == 0:
        return None, None

    min_n  = min(n_intra, len(inter_sims))
    labels = np.array([1] * min_n + [0] * min_n)
    sims   = np.array(intra_sims[:min_n] + inter_sims[:min_n])
    return labels, sims

print(f"Sample row: {df_all.iloc[0].to_dict()}")

# ── Main experiment loop ──────────────────────────────────────────────────────
results = []   # list of dicts: {algo, transform, sample_size, iteration, eer_threshold}

rng = np.random.default_rng(RANDOM_SEED)
print(f"\nRNG initialised with seed={RANDOM_SEED} — rerun with --seed {RANDOM_SEED} to reproduce these results")

for algo in TARGET_ALGOS:
    algo_mask = df_all["algo_short"].str.contains(algo, case=False)
    df_algo   = df_all[algo_mask]

    if len(df_algo) == 0:
        print(f"[WARN] No rows found for algo '{algo}' – skipping")
        continue

    # Pre-split orig rows once per algo (shared baseline for all transforms)
    df_orig_algo = (df_algo[df_algo["trans_short"].str.lower() == "orig"]
                    .set_index("filename"))
    print(f"\nalgo={algo}: {len(df_orig_algo):,} orig rows, "
          f"{df_orig_algo.index.nunique():,} unique images")

    if len(df_orig_algo) == 0:
        print(f"[WARN] No orig rows for algo '{algo}' – skipping")
        continue

    for trans in TARGET_TRANS:
        # Only rows for this specific transform (not orig)
        trans_mask = df_algo["trans_short"].str.contains(trans, case=False, na=False)
        df_trans   = df_algo[trans_mask].set_index("filename")

        if len(df_trans) == 0:
            print(f"[WARN] No rows for algo='{algo}' trans='{trans}' – skipping")
            continue

        print(f"\n=== algo={algo}  transform={trans}  "
              f"trans_rows={len(df_trans):,} ===")

        for n_img in SAMPLE_SIZES:
            desc = f"{algo}/{trans}/n={n_img:,}"
            thresholds_iter = []

            for it in tqdm(range(N_ITERATIONS), desc=desc, leave=False):
                labels, sims = build_observations(df_orig_algo, df_trans,
                                                  n_img, rng)
                if labels is None or len(np.unique(labels)) < 2:
                    continue
                try:
                    eer_t = compute_eer_threshold(labels, sims)
                    thresholds_iter.append(eer_t)
                except Exception:
                    continue

            if thresholds_iter:
                for t in thresholds_iter:
                    results.append({
                        "algo":        algo,
                        "transform":   trans,
                        "sample_size": n_img,
                        "eer_threshold": t,
                    })
                arr = np.array(thresholds_iter)
                print(f"  n={n_img:>7,}  mean={arr.mean():.4f}  std={arr.std():.4f}"
                      f"  min={arr.min():.4f}  max={arr.max():.4f}")

# ── Save raw results ──────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
csv_path = os.path.join(EXP1_OUT, "eer_thresholds_raw.csv")
df_res.to_csv(csv_path, index=False)
print(f"\nRaw results saved → {csv_path}  ({len(df_res):,} rows)")

# ── Standard deviation table (Table 1 equivalent) ────────────────────────────
pivot = (df_res.groupby(["algo", "transform", "sample_size"])["eer_threshold"]
               .std()
               .reset_index()
               .rename(columns={"eer_threshold": "std"}))
table_path = os.path.join(EXP1_OUT, "eer_std_table.csv")
pivot.to_csv(table_path, index=False)
print(f"Std-dev table saved → {table_path}")
print(pivot.to_string(index=False))

# ── Boxen-plots (Fig. 4 equivalent) ──────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)

TRANS_COLORS = {
    "Rescale":   "#4C72B0",
    "Watermark": "#DD8452",
    "Flip":      "#55A868",
}

for algo in TARGET_ALGOS:
    df_a = df_res[df_res["algo"] == algo]
    if df_a.empty:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = [TRANS_COLORS.get(t, "#999999") for t in df_a["transform"].unique()]

    sns.boxenplot(
        data=df_a,
        x="sample_size",
        y="eer_threshold",
        hue="transform",
        palette=TRANS_COLORS,
        ax=ax,
    )

    ax.set_title(f"EER Decision Threshold – {algo.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Size", fontsize=11)
    ax.set_ylabel("EER Decision Threshold (Similarity)", fontsize=11)
    actual_sizes = sorted(df_a["sample_size"].unique())
    ax.set_xticklabels([f"{s:,}" for s in actual_sizes])
    ax.legend(title="Transform", loc="best")
    plt.tight_layout()

    fig_path = os.path.join(EXP1_OUT, f"fig4_{algo}_eer_convergence.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {fig_path}")

print("\n✅  Experiment 1 complete!")
print(f"   All outputs in: {EXP1_OUT}")