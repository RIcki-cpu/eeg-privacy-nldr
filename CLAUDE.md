# EEG Privacy-Preserving NLDR — Project State

## What this project is
Research paper: "Privacy-Preserving Data Transformation Using Non-Linear Dimensionality Reduction: Balancing Privacy and Utility in Machine Learning for EEG-based BCI Systems"

Supervisor: Tuğçe Ballı. Richard is a Master's Cybersecurity student.

## Repository layout
```
stage1_features.py   ← feature extraction (run first)
stage2_baselines.py  ← PCA / UMAP / t-SNE baselines
stage3_dp.py         ← input-DP and output-DP experiments
stage4_evaluation.py ← re-ID attack, reconstruction attack, figures
config.py            ← all paths and hyperparameters (edit this only)
data/                ← .mat files (gitignored, not in repo)
outputs/             ← generated outputs (gitignored)
.venv/               ← local virtualenv (gitignored)
```

Local-only docs (not in repo): `~/Documents/NDLR_docs/`
- `PROJECT_CONTEXT.md` — full project background
- `PAPER_STRATEGY.md` — week-by-week plan
- PDFs, draft documents

## Dataset facts (confirmed)
- OpenBMI Motor Imagery, 4 subjects, binary labels: -1 (left hand), +1 (right hand)
- **100 Hz sampling rate** (confirmed via spectral analysis — alpha peak at bin 9-10)
  - Previously assumed 250 Hz in code — **this was wrong**
- 100 samples per epoch = 1.0 second per epoch
- 59 EEG channels
- (800, 59, 100) shape per subject per split
- s4_train_data_100.mat is missing (known, not provided by advisor)
- Mat keys: `dataset` (data), `labels` (labels)

## Pipeline status

| Stage | Status | Notes |
|-------|--------|-------|
| stage1_features.py | ✅ Runs locally | Fixed numpy 2.x trapz bug |
| stage2_baselines.py | ⏳ Not yet run locally | Needs umap-learn |
| stage3_dp.py | ⏳ Not yet run locally | |
| stage4_evaluation.py | ⏳ Not yet run locally | |

## Current results (from previous Colab run, see outputs/outputs/nldr/results.json)
- PCA-10D baseline: SVM=0.565 (vs 0.50 chance)
- UMAP-10D: SVM=0.495 | t-SNE-3D: SVM=0.517
- Input-DP at all ε: SVM stays flat (~0.49–0.53) — NLDR denoising effect
- Output-DP tested at ε=0.1–5.0: still flat (slope expected at ε≥50)

## Known issues fixed
- **numpy 2.x**: `np.trapz` removed → fixed with lazy `or` fallback
- **sampling rate**: was 250 Hz, now corrected to 100 Hz in config.py
- **dead features**: 177/531 were dead (delta/theta/alpha at 250 Hz assumption) → resolved by correcting sr

## Active work: 5-day strategy (see PAPER_STRATEGY.md)
- **Day 1 (current)**: Confirm sr=100 Hz ✅ → rewrite stage1 with alpha+beta+gamma (177 features) → verify 0 dead → SVM ceiling
- **Day 2**: PCA at 2/5/10/20 components, find accuracy knee
- **Day 3**: Input-DP + output-DP at chosen component count
- **Day 4**: UMAP and t-SNE comparison
- **Day 5–7**: Write results section

## How to run
```bash
source .venv/bin/activate        # or .venv/bin/python directly
pip install -r requirements.txt  # first time only
python stage1_features.py
python stage2_baselines.py
python stage3_dp.py
python stage4_evaluation.py
```
