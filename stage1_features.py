"""
STAGE 1 — Feature Extraction
═════════════════════════════════════════════════════════════════════════════
Loads OpenBMI .mat files, extracts 531-D spectral feature matrix per epoch,
drops zero-variance (dead) features, saves:

  features/features_X.npy      — (n_epochs, n_live_features)  float32
  features/features_y.npy      — (n_epochs,)                  int  {-1, 1}
  features/subject_ids.npy     — (n_epochs,)                  int  {0,1,2,…}
  features/meta.json            — dataset metadata + feature audit

CHECKPOINTING:  If all three .npy files already exist, Stage 1 is skipped
                automatically.  Set FORCE = True to recompute.

USAGE (Colab):
    # Cell 0 — mount Drive once per session:
    from google.colab import drive; drive.mount('/content/drive')
    # Cell 1 — install extras (once per session):
    !pip install -q umap-learn opacus
    # Cell 2 — edit config.py paths, then run stage:
    !python stage1_features.py

USAGE (local):
    python stage1_features.py
═════════════════════════════════════════════════════════════════════════════
"""

# ── override flag ─────────────────────────────────────────────────────────────
FORCE = False   # True = always recompute, even if outputs exist

import sys, os, re, json, warnings
import numpy as np
from scipy import signal
from scipy.io import loadmat

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as C

# ─── Checkpoint check ─────────────────────────────────────────────────────────
_outputs = [
    os.path.join(C.FEAT_DIR, 'features_X.npy'),
    os.path.join(C.FEAT_DIR, 'features_y.npy'),
    os.path.join(C.FEAT_DIR, 'subject_ids.npy'),
]
if not FORCE and all(os.path.exists(p) for p in _outputs):
    X   = np.load(_outputs[0])
    y   = np.load(_outputs[1])
    sid = np.load(_outputs[2])
    print(f"[Stage 1] Checkpoint found — skipping extraction.")
    print(f"  X shape        : {X.shape}")
    print(f"  y classes      : {np.unique(y).tolist()}")
    print(f"  Subjects       : {np.unique(sid).tolist()}")
    sys.exit(0)

# ─── Data loader ──────────────────────────────────────────────────────────────
NO_LABEL = -9999

def _to_3d(arr):
    arr = np.squeeze(np.array(arr, dtype=float))
    if arr.ndim == 3: return arr
    if arr.ndim == 2: return arr[np.newaxis]
    raise ValueError(f"Unexpected array shape: {arr.shape}")

def _load_labels(path, key_hint='labels'):
    m = loadmat(path, squeeze_me=True)
    keys = [k for k in m if not k.startswith('__')]
    key  = key_hint if key_hint in keys else next(
        (k for k in keys if any(x in k.lower() for x in ['label', 'class'])), keys[0])
    arr  = np.array(m[key]).flatten()
    if arr.dtype == object:
        arr = np.concatenate([np.array(x).flatten() for x in arr])
    return arr.astype(int), key

def scan_and_load(mat_dir, split, data_key, labels_key):
    pat_d = re.compile(r'^(?P<s>[sS]\d+)_(?P<sp>train|test)_data_(?P<sfx>[^.]+)\.mat$', re.I)
    pat_l = re.compile(r'^(?P<s>[sS]\d+)_(?P<sp>train|test)_label_(?P<sfx>[^.]+)\.mat$', re.I)
    data_files, label_lookup = [], {}

    for fname in sorted(os.listdir(mat_dir)):
        md = pat_d.match(fname)
        ml = pat_l.match(fname)
        if md:
            data_files.append({
                'path': os.path.join(mat_dir, fname),
                'subj': md.group('s').lower(),
                'split': md.group('sp').lower(),
                'sfx': md.group('sfx'),
            })
        elif ml:
            key = (ml.group('s').lower(), ml.group('sp').lower(), ml.group('sfx'))
            label_lookup[key] = os.path.join(mat_dir, fname)

    filtered = [f for f in data_files if split == 'all' or f['split'] == split.lower()]
    if not filtered:
        raise FileNotFoundError(f"No .mat files for split='{split}' in {mat_dir}")

    # Auto-detect data key if needed
    sample_keys = [k for k in loadmat(filtered[0]['path']) if not k.startswith('__')]
    if data_key not in sample_keys:
        data_key = next((k for k in sample_keys if 'data' in k.lower()), sample_keys[0])
        print(f"  data_key auto-detected: '{data_key}'")

    all_data, all_labels, all_subj_ids, n_ep_list = [], [], [], []
    for subj_idx, f in enumerate(filtered):
        mat    = loadmat(f['path'])
        epochs = _to_3d(mat[data_key])
        n_ep   = epochs.shape[0]
        all_data.append(epochs)
        n_ep_list.append(n_ep)
        all_subj_ids.append(np.full(n_ep, subj_idx, dtype=int))

        lbl_path = label_lookup.get((f['subj'], f['split'], f['sfx']))
        if lbl_path and os.path.exists(lbl_path):
            lbl, lk = _load_labels(lbl_path, labels_key)
            if len(lbl) != n_ep:
                print(f"  [WARN] {os.path.basename(lbl_path)}: {len(lbl)} labels vs {n_ep} epochs — padding")
                lbl_out = np.full(n_ep, NO_LABEL, dtype=int)
                lbl_out[:min(len(lbl), n_ep)] = lbl[:min(len(lbl), n_ep)]
                lbl = lbl_out
            print(f"  {os.path.basename(f['path'])}: {n_ep} epochs | key='{lk}' | "
                  f"classes={np.unique(lbl).tolist()}")
        else:
            lbl = np.full(n_ep, NO_LABEL, dtype=int)
            print(f"  [WARN] No label file for {os.path.basename(f['path'])}")
        all_labels.append(lbl)

    data_3d   = np.concatenate(all_data,   axis=0)
    labels    = np.concatenate(all_labels, axis=0)
    subj_ids  = np.concatenate(all_subj_ids, axis=0)

    n_ep_t, n_ch, n_t = data_3d.shape
    valid = labels != NO_LABEL
    u, c  = np.unique(labels[valid], return_counts=True)
    print(f"\n  Loaded : {n_ep_t} epochs × {n_ch} ch × {n_t} samples")
    print(f"  Classes: {u.tolist()}  counts: {c.tolist()}")
    print(f"  Missing: {(~valid).sum()} epochs (no label file)")
    print(f"  Subjects per file: {n_ep_list}")

    return data_3d, labels, subj_ids, {
        'n_epochs': n_ep_t, 'n_channels': n_ch, 'n_time': n_t,
        'subjects': sorted(set(f['subj'] for f in filtered)),
        'files': [os.path.basename(f['path']) for f in filtered],
        'n_epochs_per_file': n_ep_list,
        'sampling_rate': C.SAMPLING_RATE,
    }

# ─── Feature extraction ───────────────────────────────────────────────────────
BANDS_FE   = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
              'beta': (13, 30),  'gamma': (30, 50)}
FEAT_NAMES = ['delta_%', 'theta_%', 'alpha_%', 'beta_%', 'gamma_%',
              'centroid_hz', 'entropy_bits', 'bandwidth_95hz', 'rolloff_85hz']
_trapz     = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

def spectral_features(sig, sr):
    """
    Extract 9 spectral features from a single-channel epoch.
    Returns list of 9 floats; NaN if a band has no frequency bins.
    """
    sig    = sig - sig.mean()               # remove DC offset
    n      = len(sig)
    # nperseg calibrated so delta band (0.5 Hz lower edge) always has bins:
    # need df = sr/nperseg ≤ 0.5 Hz  →  nperseg ≥ sr/0.5 = 500 samples
    # Also cap at n//2 so Welch stays valid for short epochs.
    nperseg = min(max(int(sr / 0.5), 4), n // 2)
    f, psd  = signal.welch(sig, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, window='hann')
    total   = _trapz(psd, f) + 1e-12

    # Band-power percentages — NaN if band has no bins (handled below)
    bp = []
    for lo, hi in BANDS_FE.values():
        mask = (f >= lo) & (f <= min(hi, sr / 2 - 0.1))
        if mask.any():
            bp.append(100.0 * _trapz(psd[mask], f[mask]) / total)
        else:
            bp.append(np.nan)   # signals dead feature instead of silent zero

    centroid  = float(np.sum(f * psd) / (np.sum(psd) + 1e-12))
    p         = psd / (psd.sum() + 1e-12)
    p         = p[p > 0]
    entropy   = float(-np.sum(p * np.log2(p)))
    cum       = np.cumsum(psd / (psd.sum() + 1e-12))
    bw        = float(f[min(len(f)-1, np.searchsorted(cum, 0.975))]
                      - f[max(0, np.searchsorted(cum, 0.025) - 1)])
    rolloff   = float(f[min(len(f)-1, np.searchsorted(cum, 0.85))])

    return bp + [centroid, entropy, bw, rolloff]


# ─── Main ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 — Feature Extraction")
print("=" * 60)
print(f"\nData dir  : {C.MAT_DIR}")
print(f"Output dir: {C.OUTPUT_DIR}\n")

print("Loading .mat files...")
data_3d, labels, subj_ids, meta = scan_and_load(
    C.MAT_DIR, C.MAT_SPLIT, C.MAT_DATA_KEY, C.MAT_LABELS_KEY)

n_ep, n_ch, _ = data_3d.shape
n_feat_raw    = n_ch * len(FEAT_NAMES)
X_raw         = np.full((n_ep, n_feat_raw), np.nan, dtype=np.float32)

print(f"\nExtracting {n_ep} epochs × {n_ch} ch × {len(FEAT_NAMES)} features = {n_feat_raw}D")
print("Progress: ", end='', flush=True)
step = max(1, n_ep // 40)
for ep in range(n_ep):
    if ep % step == 0:
        print('.', end='', flush=True)
    for ch in range(n_ch):
        feats = spectral_features(data_3d[ep, ch, :], C.SAMPLING_RATE)
        s = ch * len(FEAT_NAMES)
        X_raw[ep, s:s + len(FEAT_NAMES)] = feats
print(" done.")

# ─── NaN → column median imputation ──────────────────────────────────────────
# Root cause: short epochs (100 samples, sr=250) → df=5 Hz → delta (0.5-4 Hz)
# has no Welch bins → all-NaN column → nanmedian returns NaN → imputation fails.
# Fix: replace any column whose median is NaN with 0.0 (constant → dead anyway).
n_nan = np.isnan(X_raw).sum()
if n_nan > 0:
    print(f"\nReplacing {n_nan} NaN values with column medians...")
    col_medians = np.nanmedian(X_raw, axis=0)
    # all-NaN columns have NaN median → fall back to 0.0 (will be detected as dead)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
    for j in range(X_raw.shape[1]):
        nan_rows = np.isnan(X_raw[:, j])
        if nan_rows.any():
            X_raw[nan_rows, j] = col_medians[j]
    # Final safety net: if any NaN survived (edge cases), zero them out
    still_nan = np.isnan(X_raw).sum()
    if still_nan > 0:
        print(f"  [WARN] {still_nan} NaN values remain after median imputation — zeroing out.")
        X_raw = np.where(np.isnan(X_raw), 0.0, X_raw)

# ─── Dead-feature audit ───────────────────────────────────────────────────────
# Use nanstd so that all-NaN columns (now zeroed) are caught cleanly.
# Also flag columns where std is NaN (shouldn't happen after zeroing, but defensive).
stds  = np.nanstd(X_raw, axis=0)
dead  = (stds < 1e-10) | np.isnan(stds)
n_dead = dead.sum()

dead_feat_types = {}
for idx in np.where(dead)[0]:
    feat_type = FEAT_NAMES[idx % len(FEAT_NAMES)]
    dead_feat_types[feat_type] = dead_feat_types.get(feat_type, 0) + 1

print(f"\nFeature matrix audit:")
print(f"  Shape (raw)    : {X_raw.shape}")
print(f"  NaN remaining  : {np.isnan(X_raw).sum()}")
print(f"  Dead features  : {n_dead} / {n_feat_raw}  ({100*n_dead/n_feat_raw:.1f}%)")
if dead_feat_types:
    print(f"  Dead types     : {dead_feat_types}")
    print(f"  → Dropping dead features before saving.")

# ─── Drop dead features ───────────────────────────────────────────────────────
live_mask = ~dead
X         = X_raw[:, live_mask].astype(np.float32)
live_feat_names = []
for ch in range(n_ch):
    for fi, fn in enumerate(FEAT_NAMES):
        global_idx = ch * len(FEAT_NAMES) + fi
        if live_mask[global_idx]:
            live_feat_names.append(f'ch{ch:02d}_{fn}')

print(f"  Shape (clean)  : {X.shape}  ({X.shape[1]} live features)")
print(f"  Mean           : {X.mean():.3f}")
print(f"  Std            : {X.std():.3f}")

# ─── Save ─────────────────────────────────────────────────────────────────────
np.save(os.path.join(C.FEAT_DIR, 'features_X.npy'),   X)
np.save(os.path.join(C.FEAT_DIR, 'features_y.npy'),   labels)
np.save(os.path.join(C.FEAT_DIR, 'subject_ids.npy'),  subj_ids)

meta.update({
    'n_feat_raw': int(n_feat_raw),
    'n_feat_live': int(X.shape[1]),
    'n_dead_features': int(n_dead),
    'dead_feature_types': dead_feat_types,
    'live_feature_names_sample': live_feat_names[:20],
    'label_values': np.unique(labels).tolist(),
    'n_subjects': int(len(np.unique(subj_ids))),
    'n_epochs_per_subject': {
        str(s): int((subj_ids == s).sum()) for s in np.unique(subj_ids)
    },
})
with open(os.path.join(C.FEAT_DIR, 'meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=2)

print(f"\n✓  Saved to {C.FEAT_DIR}/")
print(f"   features_X.npy    → {X.shape}")
print(f"   features_y.npy    → {labels.shape}  classes: {np.unique(labels).tolist()}")
print(f"   subject_ids.npy   → {subj_ids.shape}  subjects: {np.unique(subj_ids).tolist()}")
print(f"   meta.json         → metadata + audit")
print(f"\n→  Run stage2_baselines.py next.")
