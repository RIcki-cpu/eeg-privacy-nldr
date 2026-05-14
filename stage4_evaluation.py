"""
STAGE 4 — Evaluation, Privacy Attacks & Figures
═════════════════════════════════════════════════════════════════════════════
Loads all embeddings from Stages 2–3 and computes:

  1. Re-identification attack   — SVM classifies subject identity from embedding
                                  (chance = 1/n_subjects ≈ 33%)
  2. Reconstruction attack      — Ridge regression recovers original features
                                  from embedding (R² > 0.5 = high leakage)
  3. Summary table              — all metrics in one CSV
  4. Figures:
       fig_baselines.png         — 2D PCA projection coloured by class
       fig_tradeoff_task.png     — SVM task accuracy vs ε (input + output DP)
       fig_tradeoff_reid.png     — Re-ID accuracy vs ε (input + output DP)
       fig_reconstruction.png    — Reconstruction R² across methods

Saves:
  nldr/results_full.json    — complete metrics for all conditions
  nldr/summary_table.csv    — formatted table for the paper
  nldr/fig_*.png            — publication-quality figures (dpi=150)

USAGE:
    # Colab: !python stage4_evaluation.py
    # Local: python stage4_evaluation.py
═════════════════════════════════════════════════════════════════════════════
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # works without display (Colab/local)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import config as C

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import silhouette_score

print("=" * 60)
print("STAGE 4 — Evaluation, Privacy Attacks & Figures")
print("=" * 60)

# ─── Load features ────────────────────────────────────────────────────────────
X_all   = np.load(os.path.join(C.FEAT_DIR, 'features_X.npy'))
y_all   = np.load(os.path.join(C.FEAT_DIR, 'features_y.npy'))
sid_all = np.load(os.path.join(C.FEAT_DIR, 'subject_ids.npy'))

NO_LABEL = -9999
valid    = y_all != NO_LABEL
Xv, yv, sids = X_all[valid], y_all[valid], sid_all[valid]

classes, counts   = np.unique(yv, return_counts=True)
subjects          = np.unique(sids)
n_folds_task      = min(5, int(counts.min()))
n_folds_reid      = min(5, int(np.unique(sids, return_counts=True)[1].min()))
chance_task       = 1.0 / len(classes)
chance_reid       = 1.0 / len(subjects)
Xs                = StandardScaler().fit_transform(Xv)

cv_task = StratifiedKFold(n_splits=n_folds_task, shuffle=True, random_state=42)
cv_reid = StratifiedKFold(n_splits=n_folds_reid, shuffle=True, random_state=42)

print(f"\nFeatures : {Xv.shape}")
print(f"Classes  : {classes.tolist()}  (task chance = {chance_task:.1%})")
print(f"Subjects : {subjects.tolist()}  (re-ID chance = {chance_reid:.1%})")

# ─── Load results.json from previous stages ───────────────────────────────────
results_path = os.path.join(C.NLDR_DIR, 'results.json')
_required = [results_path,
             os.path.join(C.NLDR_DIR, 'pca_inf.npy'),
             os.path.join(C.NLDR_DIR, 'tsne_inf.npy')]
_missing  = [p for p in _required if not os.path.exists(p)]
if _missing:
    print(f"\n[ERROR] Required files from earlier stages not found:")
    for p in _missing:
        print(f"  {p}")
    print("Run stage2_baselines.py (and optionally stage3_dp.py) first.")
    sys.exit(1)

with open(results_path) as fh:
    results = json.load(fh)

# ─── Helper: load embedding ───────────────────────────────────────────────────
def load_emb(fname):
    p = os.path.join(C.NLDR_DIR, fname)
    return np.load(p) if os.path.exists(p) else None

# ─── Helper: re-ID attack ─────────────────────────────────────────────────────
def reid_attack(X_emb, subject_labels, name):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_emb)
    scores = cross_val_score(SVC(kernel='rbf', C=1, gamma='scale'),
                             X_sc, subject_labels, cv=cv_reid)
    acc    = float(scores.mean())
    leak   = "HIGH" if acc > chance_reid * 2 else ("LOW" if acc < chance_reid * 1.2 else "MEDIUM")
    print(f"  {name:40s} | Re-ID = {acc:.3f}  [{leak}]  (chance={chance_reid:.3f})")
    return round(acc, 3)

# ─── Helper: reconstruction attack ───────────────────────────────────────────
def recon_attack(X_emb, X_orig, name):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_emb)
    Xv_hat = cross_val_predict(Ridge(alpha=1.0), X_sc, X_orig, cv=5)
    ss_res = np.sum((X_orig - Xv_hat) ** 2)
    ss_tot = np.sum((X_orig - X_orig.mean(axis=0)) ** 2)
    r2     = float(1 - ss_res / ss_tot)
    mse    = float(np.mean((X_orig - Xv_hat) ** 2))
    risk   = "HIGH" if r2 > 0.5 else ("MEDIUM" if r2 > 0.1 else "LOW")
    print(f"  {name:40s} | R²={r2:.3f}  MSE={mse:.4f}  [risk:{risk}]")
    return round(r2, 3), round(mse, 4)

# ══════════════════════════════════════════════════════════════════════════════
#  RE-IDENTIFICATION ATTACK — all baseline embeddings
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"RE-IDENTIFICATION ATTACK  (SVM: who is this subject?)")
print(f"  chance level = {chance_reid:.3f}  ({1/chance_reid:.0f} subjects)")
print(f"{'─'*60}")

baseline_map = {
    'pca_inf':  f'pca_inf.npy',
    'umap_inf': f'umap_inf.npy',
    'tsne_inf': f'tsne_inf.npy',
}

# Baselines
print("\nBaselines (ε = ∞):")
for key, fname in baseline_map.items():
    emb = load_emb(fname)
    if emb is not None and key in results:
        ra = reid_attack(emb, sids, results[key]['method'])
        results[key]['reid_acc'] = ra

# Input DP
print("\nInput DP (noise before NLDR):")
for eps in C.EPSILONS_INPUT:
    for m in ['pca', 'umap', 'tsne']:
        key   = f'{m}_in_{eps}'
        fname = f'{m}_in_eps{eps}.npy'
        emb   = load_emb(fname)
        if emb is not None and key in results:
            ra = reid_attack(emb, sids, results[key]['method'])
            results[key]['reid_acc'] = ra

# Output DP
print("\nOutput DP (noise after NLDR):")
for eps in C.EPSILONS_OUTPUT:
    for m in ['pca', 'umap', 'tsne']:
        key   = f'{m}_out_{eps}'
        fname = f'{m}_out_eps{eps}.npy'
        emb   = load_emb(fname)
        if emb is not None and key in results:
            ra = reid_attack(emb, sids, results[key]['method'])
            results[key]['reid_acc'] = ra

# ══════════════════════════════════════════════════════════════════════════════
#  RECONSTRUCTION ATTACK — baseline embeddings only
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"RECONSTRUCTION ATTACK  (Ridge: embedding → original features)")
print(f"  R² > 0.5 = high leakage  |  R² < 0.1 = good privacy")
print(f"{'─'*60}")

for key, fname in baseline_map.items():
    emb = load_emb(fname)
    if emb is not None and key in results:
        r2, mse = recon_attack(emb, Xv, results[key]['method'])
        results[key]['recon_r2']  = r2
        results[key]['recon_mse'] = mse

# Also run reconstruction on DP embeddings (input mechanism)
print("\nReconstruction under input-DP (PCA only):")
for eps in C.EPSILONS_INPUT:
    key   = f'pca_in_{eps}'
    fname = f'pca_in_eps{eps}.npy'
    emb   = load_emb(fname)
    if emb is not None and key in results:
        r2, mse = recon_attack(emb, Xv, results[key]['method'])
        results[key]['recon_r2']  = r2
        results[key]['recon_mse'] = mse

print("\nReconstruction under output-DP (PCA only):")
for eps in C.EPSILONS_OUTPUT:
    key   = f'pca_out_{eps}'
    fname = f'pca_out_eps{eps}.npy'
    emb   = load_emb(fname)
    if emb is not None and key in results:
        r2, mse = recon_attack(emb, Xv, results[key]['method'])
        results[key]['recon_r2']  = r2
        results[key]['recon_mse'] = mse

# ─── Save full results ────────────────────────────────────────────────────────
full_path = os.path.join(C.NLDR_DIR, 'results_full.json')
with open(full_path, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"\n✓  results_full.json → {full_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for key, r in results.items():
    rows.append({
        'Key'             : key,
        'Method'          : r.get('method', ''),
        'ε'               : r.get('eps', ''),
        'Mechanism'       : r.get('mechanism', 'input' if '_in_' in key else
                                  'output' if '_out_' in key else 'none'),
        'N components'    : r.get('n_components', ''),
        'Trustworthiness' : r.get('trustworthiness', ''),
        'SVM Task Acc'    : r.get('svm_acc_mean', ''),
        'SVM Task ±std'   : r.get('svm_acc_std', ''),
        'Silhouette'      : r.get('silhouette', ''),
        'Re-ID Acc'       : r.get('reid_acc', ''),
        'Recon R²'        : r.get('recon_r2', ''),
    })

summary = pd.DataFrame(rows)
csv_path = os.path.join(C.NLDR_DIR, 'summary_table.csv')
summary.to_csv(csv_path, index=False)
print(f"✓  summary_table.csv → {csv_path}")
print(f"\n{summary.to_string(index=False)}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {'pca': 'steelblue', 'umap': 'coral', 'tsne': 'seagreen'}
LABELS = {'pca': 'PCA', 'umap': 'UMAP', 'tsne': 't-SNE'}

# ─── Figure 1: Baseline embeddings (PCA 2D projection for visualisation) ──────
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
cmap_c    = plt.cm.tab10(np.linspace(0, .45, len(classes)))

# Use PCA-2D projection just for visualisation (not for SVM)
pca2      = PCA(n_components=2, random_state=42)
X_pca2    = pca2.fit_transform(Xs)
ev        = pca2.explained_variance_ratio_

for ax, (key, fname, title) in zip(axes, [
    ('pca_inf',  'pca_inf.npy',  f'PCA (2D view)'),
    ('umap_inf', 'umap_inf.npy', 'UMAP (dims 1-2)'),
    ('tsne_inf', 'tsne_inf.npy', 't-SNE (dims 1-2)'),
]):
    emb = load_emb(fname)
    if emb is None:
        ax.set_visible(False)
        continue
    # Use first 2 dims for scatter
    Xe2 = emb[:, :2]
    for cls, col in zip(classes, cmap_c):
        mask = yv == cls
        ax.scatter(Xe2[mask, 0], Xe2[mask, 1], s=12, alpha=0.45,
                   color=col, label=f'Class {cls}')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')

fig.suptitle('NLDR Baseline Embeddings — No Privacy (ε=∞)  |  First 2 dims shown',
             fontweight='bold')
plt.tight_layout()
p = os.path.join(C.NLDR_DIR, 'fig_baselines.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_baselines.png")

# ─── Figure 2: Task SVM accuracy vs ε ────────────────────────────────────────
#  Panel A: Input DP  (ε = 0.1 → 5.0 → ∞)
#  Panel B: Output DP (ε = 0.1 → 100 → ∞)  ← slope appears at high ε
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

eps_in_plot  = C.EPSILONS_INPUT  + [9999]   # 9999 = visual stand-in for ∞
eps_out_plot = C.EPSILONS_OUTPUT + [9999]
eps_in_tick  = [str(e) for e in C.EPSILONS_INPUT] + ['∞']
eps_out_tick = [str(e) for e in C.EPSILONS_OUTPUT] + ['∞']

for m in ['pca', 'umap', 'tsne']:
    # Input DP
    task_in = []
    for eps in C.EPSILONS_INPUT:
        k = f'{m}_in_{eps}'
        task_in.append(results[k]['svm_acc_mean'] if k in results else np.nan)
    task_in.append(results.get(f'{m}_inf', {}).get('svm_acc_mean', np.nan))
    ax1.plot(range(len(eps_in_plot)), task_in, 'o-', lw=2, ms=6,
             color=COLORS[m], label=LABELS[m])

    # Output DP (PCA only for clarity)
    if m == 'pca':
        task_out = []
        for eps in C.EPSILONS_OUTPUT:
            k = f'{m}_out_{eps}'
            task_out.append(results[k]['svm_acc_mean'] if k in results else np.nan)
        task_out.append(results.get(f'{m}_inf', {}).get('svm_acc_mean', np.nan))
        ax2.plot(range(len(eps_out_plot)), task_out, 'o-', lw=2.5, ms=7,
                 color=COLORS[m], label=LABELS[m])

for ax, eps_tick, title in [
    (ax1, eps_in_tick,  'Task SVM Accuracy — Input DP  (noise BEFORE NLDR)'),
    (ax2, eps_out_tick, 'Task SVM Accuracy — Output DP  (noise AFTER NLDR, PCA)'),
]:
    ax.axhline(chance_task, color='gray', ls='--', lw=1.2, label=f'Chance ({chance_task:.0%})')
    ax.set_xticks(range(len(eps_tick)))
    ax.set_xticklabels(eps_tick, fontsize=9)
    ax.set_xlabel('Privacy budget ε  (← more private)', fontsize=10)
    ax.set_ylabel('SVM Task Accuracy', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_ylim(0.40, 0.72)

ax2.text(0.5, 0.93,
         'Utility slope appears at ε ≥ 50\n(task signal too weak for tight DP)',
         ha='center', transform=ax2.transAxes, fontsize=8,
         color='dimgray', style='italic')

plt.tight_layout()
p = os.path.join(C.NLDR_DIR, 'fig_tradeoff_task.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_tradeoff_task.png")

# ─── Figure 3: Re-ID accuracy vs ε (the KEY privacy figure) ──────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for m in ['pca', 'umap', 'tsne']:
    # Input DP re-ID
    reid_in = []
    for eps in C.EPSILONS_INPUT:
        k = f'{m}_in_{eps}'
        reid_in.append(results[k].get('reid_acc', np.nan) if k in results else np.nan)
    reid_in.append(results.get(f'{m}_inf', {}).get('reid_acc', np.nan))
    ax1.plot(range(len(eps_in_plot)), reid_in, 'o-', lw=2, ms=6,
             color=COLORS[m], label=LABELS[m])

    # Output DP re-ID (all methods)
    reid_out = []
    for eps in C.EPSILONS_OUTPUT:
        k = f'{m}_out_{eps}'
        reid_out.append(results[k].get('reid_acc', np.nan) if k in results else np.nan)
    reid_out.append(results.get(f'{m}_inf', {}).get('reid_acc', np.nan))
    ax2.plot(range(len(eps_out_plot)), reid_out, 'o-', lw=2, ms=6,
             color=COLORS[m], label=LABELS[m])

for ax, eps_tick, title in [
    (ax1, eps_in_tick,  'Re-Identification Accuracy — Input DP'),
    (ax2, eps_out_tick, 'Re-Identification Accuracy — Output DP'),
]:
    ax.axhline(chance_reid, color='gray', ls='--', lw=1.2,
               label=f'Chance ({chance_reid:.0%})')
    ax.set_xticks(range(len(eps_tick)))
    ax.set_xticklabels(eps_tick, fontsize=9)
    ax.set_xlabel('Privacy budget ε  (← more private)', fontsize=10)
    ax.set_ylabel('Re-ID Accuracy', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    ax.set_ylim(0.20, 1.05)

plt.suptitle('Privacy Protection — Re-Identification Attack\n'
             'Target: near chance level (good privacy)', fontweight='bold')
plt.tight_layout()
p = os.path.join(C.NLDR_DIR, 'fig_tradeoff_reid.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_tradeoff_reid.png")

# ─── Figure 4: Reconstruction R² comparison ───────────────────────────────────
methods_baseline = ['pca_inf', 'umap_inf', 'tsne_inf']
labels_b         = ['PCA-10D\n(ε=∞)', 'UMAP-10D\n(ε=∞)', 't-SNE-3D\n(ε=∞)']
r2_vals          = [results.get(k, {}).get('recon_r2', np.nan) for k in methods_baseline]

# Add PCA output-DP at ε=5 for contrast
r2_dp5 = results.get('pca_out_5.0', {}).get('recon_r2', np.nan)
r2_in1 = results.get('pca_in_1.0',  {}).get('recon_r2', np.nan)
r2_in01 = results.get('pca_in_0.1', {}).get('recon_r2', np.nan)

extra_labels = ['PCA out-DP\nε=5.0', 'PCA in-DP\nε=1.0', 'PCA in-DP\nε=0.1']
extra_vals   = [r2_dp5, r2_in1, r2_in01]
all_labels   = labels_b + extra_labels
all_vals     = r2_vals  + extra_vals
colors_bar   = ['steelblue','coral','seagreen','steelblue','steelblue','steelblue']
alphas_bar   = [0.9, 0.9, 0.9, 0.6, 0.45, 0.3]

fig, ax = plt.subplots(figsize=(11, 5))
bars    = ax.bar(range(len(all_labels)), all_vals, color=colors_bar,
                 alpha=0.85, edgecolor='white', linewidth=1.2)
for bar, alpha in zip(bars, alphas_bar):
    bar.set_alpha(alpha)

ax.axhline(0.50, color='red', ls='--', lw=1.2, label='High leakage threshold (R²=0.5)')
ax.axhline(0.10, color='green', ls='--', lw=1.2, label='Low leakage threshold (R²=0.1)')
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylabel('Reconstruction R²  (higher = more leakage)', fontsize=10)
ax.set_title('Reconstruction Attack — How Much Original Data Can Be Recovered?',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=.25, axis='y')
ax.set_ylim(-0.05, 1.05)

for i, v in enumerate(all_vals):
    if not np.isnan(v):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
p = os.path.join(C.NLDR_DIR, 'fig_reconstruction.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_reconstruction.png")

# ─── Final print ──────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"STAGE 4 COMPLETE — All outputs in {C.NLDR_DIR}/")
print(f"{'═'*60}")
print(f"\nKey findings at a glance:")
for key in ['pca_inf', 'umap_inf', 'tsne_inf']:
    if key in results:
        r = results[key]
        print(f"  {r.get('method','?'):30s}  "
              f"task={r.get('svm_acc_mean','?'):.3f}  "
              f"re-ID={r.get('reid_acc','?') if r.get('reid_acc') is not None else 'TBD'}  "
              f"R²={r.get('recon_r2','?') if r.get('recon_r2') is not None else 'TBD'}")
print(f"\n  Task chance level : {chance_task:.3f}  ({chance_task:.0%})")
print(f"  Re-ID chance level: {chance_reid:.3f}  ({chance_reid:.0%})")
print(f"\nFiles produced:")
for f in ['results_full.json', 'summary_table.csv', 'fig_baselines.png',
          'fig_tradeoff_task.png', 'fig_tradeoff_reid.png', 'fig_reconstruction.png']:
    full = os.path.join(C.NLDR_DIR, f)
    size = f"{os.path.getsize(full)//1024} KB" if os.path.exists(full) else "missing"
    print(f"  {f:35s} {size}")
