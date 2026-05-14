"""
STAGE 1 — Feature Extraction
═════════════════════════════════════════════════════════════════════════════
Loads OpenBMI .mat files, extracts alpha/beta/gamma band-power features
(3 bands × 59 channels = 177D per epoch), drops any zero-variance columns,
saves:

  features/features_X.npy    — (n_epochs, 177)  float32
  features/features_y.npy    — (n_epochs,)       int  {-1, 1}
  features/subject_ids.npy   — (n_epochs,)       int  {0, 1, 2, …}
  features/meta.json         — dataset metadata + feature audit

Then runs an SVM ceiling check on the raw 177D features.
Target: SVM accuracy > 55% (50% = chance for binary classification).

CHECKPOINTING: If all .npy files exist, extraction is skipped automatically.
               Set FORCE = True to recompute.

USAGE:
    python stage1_features.py          # local
═════════════════════════════════════════════════════════════════════════════
"""

FORCE = False   # True = always recompute, even if outputs exist

import sys, os, re, json, warnings
import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

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
    X      = np.load(_outputs[0])
    labels = np.load(_outputs[1])
    sid    = np.load(_outputs[2])
    print(f"[Stage 1] Checkpoint found — skipping extraction.")
    print(f"  X shape   : {X.shape}")
    print(f"  y classes : {np.unique(labels).tolist()}")
    print(f"  Subjects  : {np.unique(sid).tolist()}")
else:
    X = labels = sid = None  # will be set during extraction below

# ─── Data loader ──────────────────────────────────────────────────────────────
NO_LABEL = -9999

def _to_3d(arr):
    arr = np.squeeze(np.array(arr, dtype=float))
    if arr.ndim == 3: return arr
    if arr.ndim == 2: return arr[np.newaxis]
    raise ValueError(f"Unexpected array shape: {arr.shape}")

def _load_labels(path, key_hint='labels'):
    m    = loadmat(path, squeeze_me=True)
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

    data_3d  = np.concatenate(all_data,    axis=0)
    labels_  = np.concatenate(all_labels,  axis=0)
    subj_ids = np.concatenate(all_subj_ids, axis=0)

    n_ep_t, n_ch, n_t = data_3d.shape
    valid = labels_ != NO_LABEL
    u, c  = np.unique(labels_[valid], return_counts=True)
    print(f"\n  Loaded : {n_ep_t} epochs × {n_ch} ch × {n_t} samples")
    print(f"  sr     : {C.SAMPLING_RATE} Hz  →  epoch = {n_t/C.SAMPLING_RATE:.2f} s")
    print(f"  Classes: {u.tolist()}  counts: {c.tolist()}")
    print(f"  Missing: {(~valid).sum()} epochs (no label file)")
    print(f"  Subjects per file: {n_ep_list}")

    return data_3d, labels_, subj_ids, {
        'n_epochs': n_ep_t, 'n_channels': n_ch, 'n_time': n_t,
        'sampling_rate': C.SAMPLING_RATE,
        'subjects': sorted(set(f['subj'] for f in filtered)),
        'files': [os.path.basename(f['path']) for f in filtered],
        'n_epochs_per_file': n_ep_list,
    }

# ─── Feature extraction — alpha / beta / gamma band power only ────────────────
# With sr=100 Hz, n=100, nperseg=50:
#   df = 2 Hz  →  alpha (8-13 Hz): bins 4-6  ✓
#                  beta  (13-30 Hz): bins 7-15 ✓
#                  gamma (30-50 Hz): bins 15-25 ✓
# Expected dead features: 0

BANDS_FE   = {'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
FEAT_NAMES = ['alpha_%', 'beta_%', 'gamma_%']
_trapz     = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

def spectral_features(sig, sr):
    """Band-power % for alpha, beta, gamma. Returns list of 3 floats."""
    sig     = sig - sig.mean()                   # remove DC offset
    n       = len(sig)
    nperseg = min(max(int(sr / 0.5), 4), n // 2) # 50 at sr=100, n=100
    f, psd  = signal.welch(sig, fs=sr, nperseg=nperseg,
                            noverlap=nperseg // 2, window='hann')
    total   = _trapz(psd, f) + 1e-12

    bp = []
    for lo, hi in BANDS_FE.values():
        mask = (f >= lo) & (f <= min(hi, sr / 2 - 0.1))
        bp.append(100.0 * _trapz(psd[mask], f[mask]) / total if mask.any() else np.nan)
    return bp

# ─── Main ─────────────────────────────────────────────────────────────────────
if X is None:   # not loaded from checkpoint
    print("=" * 60)
    print("STAGE 1 — Feature Extraction  (alpha/beta/gamma, 177D)")
    print("=" * 60)
    print(f"\nData dir  : {C.MAT_DIR}")
    print(f"Output dir: {C.OUTPUT_DIR}\n")

    print("Loading .mat files...")
    data_3d, labels, sid, meta = scan_and_load(
        C.MAT_DIR, C.MAT_SPLIT, C.MAT_DATA_KEY, C.MAT_LABELS_KEY)

    n_ep, n_ch, _ = data_3d.shape
    n_feat_raw    = n_ch * len(FEAT_NAMES)
    X_raw         = np.full((n_ep, n_feat_raw), np.nan, dtype=np.float32)

    print(f"\nExtracting {n_ep} epochs × {n_ch} ch × {len(FEAT_NAMES)} bands = {n_feat_raw}D")
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

    # ─── NaN imputation ───────────────────────────────────────────────────────
    n_nan = int(np.isnan(X_raw).sum())
    if n_nan > 0:
        print(f"\nReplacing {n_nan} NaN values with column medians...")
        col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        for j in range(X_raw.shape[1]):
            nan_rows = np.isnan(X_raw[:, j])
            if nan_rows.any():
                X_raw[nan_rows, j] = col_medians[j]
        still_nan = int(np.isnan(X_raw).sum())
        if still_nan > 0:
            print(f"  [WARN] {still_nan} NaN remain — zeroing out.")
            X_raw = np.where(np.isnan(X_raw), 0.0, X_raw)

    # ─── Dead-feature audit ───────────────────────────────────────────────────
    stds  = np.nanstd(X_raw, axis=0)
    dead  = (stds < 1e-10) | np.isnan(stds)
    n_dead = int(dead.sum())

    dead_types = {}
    for idx in np.where(dead)[0]:
        ft = FEAT_NAMES[idx % len(FEAT_NAMES)]
        dead_types[ft] = dead_types.get(ft, 0) + 1

    print(f"\nFeature matrix audit:")
    print(f"  Shape (raw)  : {X_raw.shape}")
    print(f"  NaN remaining: {int(np.isnan(X_raw).sum())}")
    print(f"  Dead features: {n_dead} / {n_feat_raw}", end="")
    if n_dead == 0:
        print("  ✓  (zero dead — all bands resolvable at sr=100 Hz)")
    else:
        print(f"\n  Dead types   : {dead_types}")

    live_mask = ~dead
    X         = X_raw[:, live_mask].astype(np.float32)
    live_names = [f'ch{ch:02d}_{fn}'
                  for ch in range(n_ch)
                  for fi, fn in enumerate(FEAT_NAMES)
                  if live_mask[ch * len(FEAT_NAMES) + fi]]

    print(f"  Shape (clean): {X.shape}")
    print(f"  Mean / Std   : {X.mean():.3f} / {X.std():.3f}")

    # ─── Save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(C.FEAT_DIR, 'features_X.npy'),  X)
    np.save(os.path.join(C.FEAT_DIR, 'features_y.npy'),  labels)
    np.save(os.path.join(C.FEAT_DIR, 'subject_ids.npy'), sid)

    meta.update({
        'feature_set'       : 'alpha_beta_gamma_band_power',
        'bands'             : {k: list(v) for k, v in BANDS_FE.items()},
        'n_feat_raw'        : int(n_feat_raw),
        'n_feat_live'       : int(X.shape[1]),
        'n_dead_features'   : n_dead,
        'dead_feature_types': dead_types,
        'label_values'      : np.unique(labels).tolist(),
        'n_subjects'        : int(len(np.unique(sid))),
    })
    with open(os.path.join(C.FEAT_DIR, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n✓  Saved to {C.FEAT_DIR}/")
    print(f"   features_X.npy  → {X.shape}")
    print(f"   features_y.npy  → {labels.shape}  classes: {np.unique(labels).tolist()}")

# ─── SVM Ceiling — raw features (no compression) ─────────────────────────────
print("\n" + "=" * 60)
print("SVM CEILING — Raw Features (no dimensionality reduction)")
print("=" * 60)

valid   = labels != NO_LABEL
Xv, yv  = X[valid], labels[valid]
Xs      = StandardScaler().fit_transform(Xv)
sid_v   = sid[valid]

classes, counts = np.unique(yv, return_counts=True)
n_folds = min(5, int(counts.min()))
chance  = 1.0 / len(classes)
cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\n  Features : {Xs.shape[1]}D  |  epochs: {len(yv)}")
print(f"  Classes  : {classes.tolist()}  counts: {counts.tolist()}")
print(f"  Chance   : {chance:.3f}  ({chance:.1%})")
print(f"  CV folds : {n_folds}\n")

# All subjects
scores_all = cross_val_score(SVC(kernel='rbf', C=1, gamma='scale'), Xs, yv, cv=cv)
mean_all   = scores_all.mean()
flag       = "✓ PROCEED" if mean_all > 0.55 else "✗ BELOW THRESHOLD — investigate"
print(f"  All subjects  SVM : {mean_all:.3f} ± {scores_all.std():.3f}  [{flag}]")

# Per-subject breakdown
print(f"\n  Per-subject breakdown:")
for s_idx in np.unique(sid_v):
    mask = sid_v == s_idx
    if mask.sum() < 10:
        continue
    Xs_s = StandardScaler().fit_transform(Xv[mask])
    yv_s = yv[mask]
    cls, cnt = np.unique(yv_s, return_counts=True)
    nf  = min(5, int(cnt.min()))
    sc  = cross_val_score(SVC(kernel='rbf', C=1, gamma='scale'), Xs_s, yv_s,
                          cv=StratifiedKFold(n_splits=nf, shuffle=True, random_state=42))
    print(f"    Subject {s_idx} : {sc.mean():.3f} ± {sc.std():.3f}  (n={mask.sum()})")

print(f"\n→  Run stage2_baselines.py next.")
