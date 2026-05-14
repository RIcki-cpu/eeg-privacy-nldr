# ══════════════════════════════════════════════════════════════════════════════
#  EEG Privacy-Preserving NLDR  —  Pipeline Configuration
#  Edit ONLY this file, then run stages in order.
# ══════════════════════════════════════════════════════════════════════════════

import os

# ─── PATHS ────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))

MAT_DIR    = os.path.join(_BASE, 'data')     # folder with .mat files
OUTPUT_DIR = os.path.join(_BASE, 'outputs')  # where figures + features go

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
SAMPLING_RATE  = 250       # Hz
MAT_SPLIT      = 'train'   # 'train' | 'test' | 'all'
MAT_DATA_KEY   = 'dataset' # key inside data .mat files
MAT_LABELS_KEY = 'labels'  # key inside label .mat files  (NOT 'classlabel')

# ─── NLDR HYPERPARAMETERS ─────────────────────────────────────────────────────
N_COMPONENTS_PCA  = 10   # PCA latent dimensions for SVM evaluation
N_COMPONENTS_UMAP = 10   # UMAP latent dimensions
N_COMPONENTS_TSNE = 3    # t-SNE (rarely useful >3; 3D is the safe max)
UMAP_N_NEIGHBORS  = 15   # UMAP neighbourhood size

# ─── DIFFERENTIAL PRIVACY ─────────────────────────────────────────────────────
DP_DELTA          = 1e-5  # δ parameter (standard for (ε,δ)-DP)

# ε values for INPUT perturbation (noise before NLDR)
EPSILONS_INPUT  = [0.1, 0.5, 1.0, 5.0]

# ε values for OUTPUT perturbation (noise after NLDR, on the embedding)
# Extended range so the utility-privacy slope is visible.
# Re-ID slope appears at ε ≤ 5; task-accuracy slope appears at ε ≥ 50.
EPSILONS_OUTPUT = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# ─── DERIVED PATHS (do not edit) ──────────────────────────────────────────────
FEAT_DIR  = os.path.join(OUTPUT_DIR, 'features')
NLDR_DIR  = os.path.join(OUTPUT_DIR, 'nldr')
ANAL_DIR  = os.path.join(OUTPUT_DIR, 'analysis')

for _d in [OUTPUT_DIR, FEAT_DIR, NLDR_DIR, ANAL_DIR]:
    os.makedirs(_d, exist_ok=True)
