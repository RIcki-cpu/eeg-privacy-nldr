"""
STAGE 2 — NLDR Baselines (no privacy)
═════════════════════════════════════════════════════════════════════════════
Runs PCA, UMAP, and t-SNE on the clean feature matrix from Stage 1.
Evaluates each embedding with SVM (5-fold CV) and trustworthiness.
Also prints SVM on raw features as the reference ceiling.

Saves:
  nldr/pca_inf.npy   — PCA-{N}D embedding
  nldr/umap_inf.npy  — UMAP-{N}D embedding
  nldr/tsne_inf.npy  — t-SNE-{N}D embedding
  nldr/results.json  — metrics dict (grows across stages)

CHECKPOINTING: Skips any embedding whose .npy file already exists with the
               correct shape. Set FORCE = True to recompute all.

USAGE:
    # Colab: !python stage2_baselines.py
    # Local: python stage2_baselines.py
═════════════════════════════════════════════════════════════════════════════
"""

FORCE = False   # True = recompute even if files exist

import sys, os, json, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as C

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score

try:
    from sklearn.manifold import trustworthiness as _trust
    HAS_TRUST = True
except ImportError:
    HAS_TRUST = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed. Install with: pip install umap-learn")

print("=" * 60)
print("STAGE 2 — NLDR Baselines")
print("=" * 60)

# ─── Load features ────────────────────────────────────────────────────────────
X_path   = os.path.join(C.FEAT_DIR, 'features_X.npy')
y_path   = os.path.join(C.FEAT_DIR, 'features_y.npy')
if not (os.path.exists(X_path) and os.path.exists(y_path)):
    print("[ERROR] Feature files not found. Run stage1_features.py first.")
    sys.exit(1)

X_all = np.load(X_path)
y_all = np.load(y_path)

# Defensive: catch any NaN/Inf that slipped through Stage 1 (e.g. all-NaN
# columns whose median was also NaN and thus zeroing was missed).
bad = ~np.isfinite(X_all)
if bad.any():
    n_bad = int(bad.sum())
    print(f"[WARN] {n_bad} NaN/Inf values found in features_X.npy — zeroing out.")
    print(f"       Re-run stage1_features.py with FORCE=True to fix at source.")
    X_all = np.where(np.isfinite(X_all), X_all, 0.0)

# Filter to labelled epochs only
NO_LABEL = -9999
valid    = y_all != NO_LABEL
Xv, yv   = X_all[valid], y_all[valid]
classes, counts = np.unique(yv, return_counts=True)
n_folds  = min(5, int(counts.min()))
chance   = 1.0 / len(classes)

print(f"\nFeatures loaded : {Xv.shape}  (dead features already removed in Stage 1)")
print(f"Classes         : {classes.tolist()}  counts: {counts.tolist()}")
print(f"Chance level    : {chance:.1%}")
print(f"CV folds        : {n_folds}")

# ─── Standardise ─────────────────────────────────────────────────────────────
Xs = StandardScaler().fit_transform(Xv)

cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# ─── Reference ceiling: SVM on raw features ───────────────────────────────────
print(f"\nReference ceiling — SVM on raw {Xs.shape[1]}D features:")
raw_scores = cross_val_score(SVC(kernel='rbf', C=1, gamma='scale'), Xs, yv, cv=cv)
print(f"  SVM {Xs.shape[1]}D  : {raw_scores.mean():.3f} ± {raw_scores.std():.3f}  "
      f"(ceiling — target for NLDR to approach)")

# PCA variance curve
pca_full = PCA().fit(Xs)
for n in [2, 5, 10, 20, 50]:
    ev = pca_full.explained_variance_ratio_[:n].sum()
    print(f"  PCA-{n:3d} var exp: {ev:.3f}")

# ─── Load existing results ────────────────────────────────────────────────────
results_path = os.path.join(C.NLDR_DIR, 'results.json')
if os.path.exists(results_path):
    with open(results_path) as fh:
        results = json.load(fh)
else:
    results = {}

# ─── Shared evaluation function ───────────────────────────────────────────────
def evaluate(X_emb, y_true, name, X_orig, eps_val='inf'):
    """Compute trustworthiness, SVM acc, silhouette. Returns metrics dict."""
    # Trustworthiness: only for ≤ 5000 samples (slow for large n)
    if HAS_TRUST and len(X_orig) <= 5000:
        T = float(_trust(X_orig, X_emb, n_neighbors=5))
    else:
        T = None

    svc    = SVC(kernel='rbf', C=1, gamma='scale')
    scores = cross_val_score(svc, X_emb, y_true, cv=cv)
    sil    = float(silhouette_score(X_emb, y_true)) if len(np.unique(y_true)) > 1 else None

    T_s = f"{T:.3f}" if T is not None else "n/a"
    print(f"  {name:30s} | T={T_s} | SVM={scores.mean():.3f}±{scores.std():.3f} | "
          f"Sil={sil:.3f if sil is not None else 'n/a'}")
    return {
        'method'         : name,
        'eps'            : str(eps_val),
        'n_components'   : int(X_emb.shape[1]),
        'trustworthiness': round(T, 3) if T is not None else None,
        'svm_acc_mean'   : round(float(scores.mean()), 3),
        'svm_acc_std'    : round(float(scores.std()),  3),
        'silhouette'     : round(sil, 3) if sil is not None else None,
    }

# ─── PCA ──────────────────────────────────────────────────────────────────────
print(f"\nNLDR Baselines  (ε = ∞, no privacy)\n" + "─" * 60)

X_pca = None  # initialise so the second guard never hits NameError
pca_path = os.path.join(C.NLDR_DIR, 'pca_inf.npy')
if not FORCE and os.path.exists(pca_path):
    _tmp = np.load(pca_path)
    if _tmp.shape == (len(Xv), C.N_COMPONENTS_PCA):
        X_pca = _tmp
        print(f"  PCA-{C.N_COMPONENTS_PCA}D   : loaded from checkpoint")
        results['pca_inf'] = evaluate(X_pca, yv, f'PCA-{C.N_COMPONENTS_PCA}D (ε=∞)', Xs)
    else:
        print(f"  PCA checkpoint shape mismatch {_tmp.shape} → expected "
              f"({len(Xv)},{C.N_COMPONENTS_PCA}) — recomputing...")

if X_pca is None:
    pca   = PCA(n_components=C.N_COMPONENTS_PCA, random_state=42)
    X_pca = pca.fit_transform(Xs)
    ev    = pca.explained_variance_ratio_.sum()
    print(f"  PCA-{C.N_COMPONENTS_PCA}D   : fitted  (explained var = {ev:.3f})")
    np.save(pca_path, X_pca)
    results['pca_inf'] = evaluate(X_pca, yv, f'PCA-{C.N_COMPONENTS_PCA}D (ε=∞)', Xs)

# ─── UMAP ─────────────────────────────────────────────────────────────────────
X_umap = None  # initialise
umap_path = os.path.join(C.NLDR_DIR, 'umap_inf.npy')
if not HAS_UMAP:
    print("\n  UMAP skipped (umap-learn not installed)")
else:
    if not FORCE and os.path.exists(umap_path):
        _tmp = np.load(umap_path)
        if _tmp.shape == (len(Xv), C.N_COMPONENTS_UMAP):
            X_umap = _tmp
            print(f"  UMAP-{C.N_COMPONENTS_UMAP}D  : loaded from checkpoint")
            results['umap_inf'] = evaluate(X_umap, yv, f'UMAP-{C.N_COMPONENTS_UMAP}D (ε=∞)', Xs)
        else:
            print(f"  UMAP checkpoint shape mismatch — recomputing...")

    if X_umap is None:
        print(f"\n  UMAP-{C.N_COMPONENTS_UMAP}D  : fitting (~1-2 min)...")
        X_umap = umap.UMAP(
            n_components=C.N_COMPONENTS_UMAP,
            n_neighbors=C.UMAP_N_NEIGHBORS,
            random_state=42,
        ).fit_transform(Xs)
        np.save(umap_path, X_umap)
        results['umap_inf'] = evaluate(X_umap, yv, f'UMAP-{C.N_COMPONENTS_UMAP}D (ε=∞)', Xs)

# ─── t-SNE ────────────────────────────────────────────────────────────────────
X_tsne = None  # initialise
tsne_path = os.path.join(C.NLDR_DIR, 'tsne_inf.npy')
if not FORCE and os.path.exists(tsne_path):
    _tmp = np.load(tsne_path)
    if _tmp.shape == (len(Xv), C.N_COMPONENTS_TSNE):
        X_tsne = _tmp
        print(f"  t-SNE-{C.N_COMPONENTS_TSNE}D : loaded from checkpoint")
        results['tsne_inf'] = evaluate(X_tsne, yv, f't-SNE-{C.N_COMPONENTS_TSNE}D (ε=∞)', Xs)
    else:
        print(f"  t-SNE checkpoint shape mismatch — recomputing...")

if X_tsne is None:
    perp   = min(30, max(5, len(Xv) // 4))
    print(f"\n  t-SNE-{C.N_COMPONENTS_TSNE}D : fitting (perplexity={perp}, ~2-3 min)...")
    X_tsne = TSNE(
        n_components=C.N_COMPONENTS_TSNE,
        perplexity=perp,
        random_state=42,
    ).fit_transform(Xs)
    np.save(tsne_path, X_tsne)
    results['tsne_inf'] = evaluate(X_tsne, yv, f't-SNE-{C.N_COMPONENTS_TSNE}D (ε=∞)', Xs)

# ─── Save results ─────────────────────────────────────────────────────────────
with open(results_path, 'w') as fh:
    json.dump(results, fh, indent=2)

print(f"\n✓  results.json updated → {results_path}")
print(f"\nBaseline summary:")
print(f"  {'Method':30s} | {'SVM Acc':>9} | {'vs chance':>10}")
print("  " + "─" * 55)
for key in ['pca_inf', 'umap_inf', 'tsne_inf']:
    if key in results:
        r   = results[key]
        acc = r['svm_acc_mean']
        gap = acc - chance
        print(f"  {r['method']:30s} | {acc:>9.3f} | +{gap:.3f} vs {chance:.3f}")

print(f"\n→  Run stage3_dp.py next.")
