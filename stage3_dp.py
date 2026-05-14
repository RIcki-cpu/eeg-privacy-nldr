"""
STAGE 3 — Differential Privacy Experiments
═════════════════════════════════════════════════════════════════════════════
Runs TWO DP mechanisms on PCA, UMAP, and t-SNE embeddings:

  INPUT perturbation  — Gaussian noise added to the 531D feature matrix
                        BEFORE NLDR runs. Tests the "NLDR denoising" hypothesis.
                        ε ∈ {0.1, 0.5, 1.0, 5.0}

  OUTPUT perturbation — Gaussian noise added to the embedding AFTER NLDR.
                        Extended ε range to show utility-privacy slope.
                        ε ∈ {0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0}

Saves:
  nldr/{method}_in_eps{ε}.npy   — input-DP embeddings
  nldr/{method}_out_eps{ε}.npy  — output-DP embeddings (from baseline)
  nldr/results.json              — updated with all DP metrics

CHECKPOINTING: Each (method, ε) pair skips if the .npy file already exists.
               Set FORCE = True to recompute everything.

USAGE:
    # Colab: !python stage3_dp.py
    # Local: python stage3_dp.py
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

print("=" * 60)
print("STAGE 3 — Differential Privacy Experiments")
print("=" * 60)

# ─── Load features ────────────────────────────────────────────────────────────
X_all = np.load(os.path.join(C.FEAT_DIR, 'features_X.npy'))
y_all = np.load(os.path.join(C.FEAT_DIR, 'features_y.npy'))
NO_LABEL = -9999
valid    = y_all != NO_LABEL
Xv, yv   = X_all[valid], y_all[valid]
classes, counts = np.unique(yv, return_counts=True)
n_folds  = min(5, int(counts.min()))
Xs       = StandardScaler().fit_transform(Xv)
cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
chance   = 1.0 / len(classes)

print(f"\nFeatures: {Xs.shape}  |  classes: {classes.tolist()}  |  chance: {chance:.1%}")

# ─── Load baseline embeddings (require Stage 2 to have run) ──────────────────
_required = [os.path.join(C.NLDR_DIR, 'pca_inf.npy'),
             os.path.join(C.NLDR_DIR, 'tsne_inf.npy')]
_missing  = [p for p in _required if not os.path.exists(p)]
if _missing:
    print(f"\n[ERROR] Baseline embeddings not found:")
    for p in _missing:
        print(f"  {p}")
    print("Run stage2_baselines.py first, then re-run this script.")
    sys.exit(1)

pca_inf  = np.load(os.path.join(C.NLDR_DIR, 'pca_inf.npy'))
umap_path = os.path.join(C.NLDR_DIR, 'umap_inf.npy')
umap_inf = np.load(umap_path) if (HAS_UMAP and os.path.exists(umap_path)) else None
tsne_inf = np.load(os.path.join(C.NLDR_DIR, 'tsne_inf.npy'))

# ─── Load existing results ────────────────────────────────────────────────────
results_path = os.path.join(C.NLDR_DIR, 'results.json')
if os.path.exists(results_path):
    with open(results_path) as fh:
        results = json.load(fh)
else:
    results = {}

# ─── Shared evaluation ───────────────────────────────────────────────────────
def evaluate(X_emb, y_true, name, X_orig, eps_val):
    if HAS_TRUST and len(X_orig) <= 5000:
        T = float(_trust(X_orig, X_emb, n_neighbors=5))
    else:
        T = None
    scores = cross_val_score(SVC(kernel='rbf', C=1, gamma='scale'), X_emb, y_true, cv=cv)
    sil    = float(silhouette_score(X_emb, y_true)) if len(np.unique(y_true)) > 1 else None
    T_s    = f"{T:.3f}" if T is not None else "n/a"
    print(f"    {name:35s} | T={T_s} | SVM={scores.mean():.3f}±{scores.std():.3f}")
    return {
        'method'         : name,
        'eps'            : str(eps_val),
        'n_components'   : int(X_emb.shape[1]),
        'trustworthiness': round(T, 3) if T is not None else None,
        'svm_acc_mean'   : round(float(scores.mean()), 3),
        'svm_acc_std'    : round(float(scores.std()),  3),
        'silhouette'     : round(sil, 3) if sil is not None else None,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  MECHANISM 1 — INPUT PERTURBATION
#  Noise added to the feature matrix BEFORE NLDR runs.
#  Hypothesis: NLDR denoises the noise (confirmed when SVM stays near chance).
# ══════════════════════════════════════════════════════════════════════════════

def input_dp(X_scaled, epsilon, delta=C.DP_DELTA, clip_norm=3.0):
    """
    Gaussian mechanism for input perturbation.
    σ = clip_norm × √(2 ln(1.25/δ)) / ε
    clip_norm=3.0 matches StandardScaler output range.
    """
    sigma     = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    X_clipped = np.clip(X_scaled, -clip_norm, clip_norm)
    return X_clipped + np.random.normal(0, sigma, X_clipped.shape), sigma

print(f"\n{'─'*60}")
print("MECHANISM 1 — Input Perturbation  (noise BEFORE NLDR)")
print(f"  ε values: {C.EPSILONS_INPUT}")
print(f"{'─'*60}")

for eps in C.EPSILONS_INPUT:
    np.random.seed(42)
    X_noisy, sigma = input_dp(Xs, eps)
    print(f"\n  ε = {eps}  (σ = {sigma:.2f})")

    # ── PCA input DP ──
    key = f'pca_in_{eps}'
    fpath = os.path.join(C.NLDR_DIR, f'pca_in_eps{eps}.npy')
    if not FORCE and os.path.exists(fpath):
        X_e = np.load(fpath)
        print(f"    PCA input-DP ε={eps}  : loaded from checkpoint")
    else:
        X_e = PCA(n_components=C.N_COMPONENTS_PCA, random_state=42).fit_transform(X_noisy)
        np.save(fpath, X_e)
    results[key] = evaluate(X_e, yv, f'PCA input-DP ε={eps}', Xs, eps)

    # ── UMAP input DP ──
    if HAS_UMAP:
        key = f'umap_in_{eps}'
        fpath = os.path.join(C.NLDR_DIR, f'umap_in_eps{eps}.npy')
        if not FORCE and os.path.exists(fpath):
            X_e = np.load(fpath)
            print(f"    UMAP input-DP ε={eps} : loaded from checkpoint")
        else:
            print(f"    UMAP input-DP ε={eps} : fitting...")
            X_e = umap.UMAP(
                n_components=C.N_COMPONENTS_UMAP,
                n_neighbors=C.UMAP_N_NEIGHBORS,
                random_state=42,
            ).fit_transform(X_noisy)
            np.save(fpath, X_e)
        results[key] = evaluate(X_e, yv, f'UMAP input-DP ε={eps}', Xs, eps)

    # ── t-SNE input DP ──
    key = f'tsne_in_{eps}'
    fpath = os.path.join(C.NLDR_DIR, f'tsne_in_eps{eps}.npy')
    if not FORCE and os.path.exists(fpath):
        X_e = np.load(fpath)
        print(f"    t-SNE input-DP ε={eps}: loaded from checkpoint")
    else:
        perp = min(30, max(5, len(Xv) // 4))
        X_e  = TSNE(n_components=C.N_COMPONENTS_TSNE, perplexity=perp,
                    random_state=42).fit_transform(X_noisy)
        np.save(fpath, X_e)
    results[key] = evaluate(X_e, yv, f't-SNE input-DP ε={eps}', Xs, eps)

# Intermediate save
with open(results_path, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n  [Intermediate save] results.json updated after input-DP experiments.")

# ══════════════════════════════════════════════════════════════════════════════
#  MECHANISM 2 — OUTPUT PERTURBATION
#  Noise added AFTER NLDR, directly on the embedding.
#  Sensitivity: empirical 99th-percentile L2 row-norm of baseline embedding.
#  This avoids the "denoising problem" — noise goes into the SVM's input space.
#
#  KEY INSIGHT on the extended ε range:
#    The PCA-10D class separation in normalized space ≈ 0.020.
#    Sigma = sensitivity × √(2 ln(1.25/δ)) / ε
#    For noise to equal class separation: ε ≈ sensitivity × 4.84 / 0.020 ≈ 200–500
#    → Task-accuracy slope appears at high ε (50–100): "weak privacy, some utility"
#    → Re-ID slope appears at low ε (1–5): "strong privacy, re-ID near chance"
#    → This gap between the two slopes is itself the key finding.
# ══════════════════════════════════════════════════════════════════════════════

def output_dp(X_emb_base, epsilon, delta=C.DP_DELTA):
    """
    Output perturbation on a baseline embedding.
    Sensitivity = 99th-percentile L2 row-norm (empirical, tighter than worst-case).
    Rows are L2-clipped to this value before noise is added.
    """
    row_norms   = np.linalg.norm(X_emb_base, axis=1, keepdims=True)
    sensitivity = float(np.percentile(row_norms, 99))
    # Clip rows to sensitivity ball
    scale       = np.minimum(1.0, sensitivity / (row_norms + 1e-10))
    X_clipped   = X_emb_base * scale
    # Gaussian mechanism
    sigma       = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise       = np.random.normal(0, sigma, X_clipped.shape)
    return X_clipped + noise, sigma, sensitivity

print(f"\n{'─'*60}")
print("MECHANISM 2 — Output Perturbation  (noise AFTER NLDR on embedding)")
print(f"  ε values: {C.EPSILONS_OUTPUT}")
print(f"  Note: task-utility slope expected at ε ≥ 50 (re-ID slope at ε ≤ 5)")
print(f"{'─'*60}")

for method_name, baseline_emb, method_key in [
    ('PCA',  pca_inf,  'pca'),
    ('UMAP', umap_inf, 'umap') if HAS_UMAP else (None, None, None),
    ('t-SNE', tsne_inf, 'tsne'),
]:
    if method_name is None or baseline_emb is None:
        continue

    # Compute sensitivity once per method
    row_norms   = np.linalg.norm(baseline_emb, axis=1)
    sensitivity = float(np.percentile(row_norms, 99))
    n_d         = baseline_emb.shape[1]
    print(f"\n  {method_name}-{n_d}D  (empirical sensitivity = {sensitivity:.4f})")

    for eps in C.EPSILONS_OUTPUT:
        key   = f'{method_key}_out_{eps}'
        fpath = os.path.join(C.NLDR_DIR, f'{method_key}_out_eps{eps}.npy')

        if not FORCE and os.path.exists(fpath):
            X_e = np.load(fpath)
            print(f"    output-DP ε={eps:5.1f}: loaded from checkpoint")
        else:
            np.random.seed(42)
            X_e, sigma, _ = output_dp(baseline_emb, eps)
            np.save(fpath, X_e)
            print(f"    output-DP ε={eps:5.1f}: σ={sigma:.4f}", end='')
            # Quick SNR estimate in embedding space
            signal_var = baseline_emb.var()
            snr_db     = 10 * np.log10(signal_var / sigma**2) if sigma > 0 else float('inf')
            print(f"  SNR={snr_db:.1f} dB")

        scores = cross_val_score(
            SVC(kernel='rbf', C=1, gamma='scale'), X_e, yv, cv=cv)
        results[key] = {
            'method'      : f'{method_name} output-DP ε={eps}',
            'eps'         : str(eps),
            'mechanism'   : 'output',
            'n_components': int(X_e.shape[1]),
            'svm_acc_mean': round(float(scores.mean()), 3),
            'svm_acc_std' : round(float(scores.std()),  3),
            'sensitivity' : round(sensitivity, 6),
        }

# ─── Final save ───────────────────────────────────────────────────────────────
with open(results_path, 'w') as fh:
    json.dump(results, fh, indent=2)

print(f"\n✓  results.json saved → {results_path}")
print(f"\nInput-DP task SVM summary (mechanism 1):")
print(f"  {'Key':25s} | {'SVM Acc':>9}")
for eps in C.EPSILONS_INPUT:
    for m in ['pca', 'umap', 'tsne']:
        k = f'{m}_in_{eps}'
        if k in results:
            print(f"  {k:25s} | {results[k]['svm_acc_mean']:>9.3f}")

print(f"\nOutput-DP task SVM summary (mechanism 2, PCA only):")
print(f"  {'ε':>6} | {'SVM Acc':>9}  ← slope should appear at ε ≥ 50")
for eps in C.EPSILONS_OUTPUT:
    k = f'pca_out_{eps}'
    if k in results:
        print(f"  {eps:>6} | {results[k]['svm_acc_mean']:>9.3f}")

print(f"\n→  Run stage4_evaluation.py next.")
