# EEG Privacy-Preserving NLDR

Privacy-preserving data transformation using non-linear dimensionality reduction (NLDR) for EEG-based BCI systems. Compares PCA, UMAP, t-SNE, and Autoencoder under (ε,δ)-Differential Privacy, evaluating the utility–privacy trade-off via SVM classification accuracy, re-identification attacks, and reconstruction attacks.

## Dataset

OpenBMI Motor Imagery dataset — 4 subjects, binary labels (left/right hand), 59 EEG channels, 250 Hz. Place `.mat` files in the `data/` directory (not included in repo).

Expected structure:
```
data/
  s1_train_data_100.mat   s1_train_label_100.mat
  s2_train_data_100.mat   s2_train_label_100.mat
  s3_train_data_100.mat   s3_train_label_100.mat
  s1_test_data_100.mat    s1_test_label_100.mat
  ...
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

Run stages in order:

```bash
python stage1_features.py   # extract 531D spectral features from .mat files
python stage2_baselines.py  # PCA / UMAP / t-SNE baselines (no privacy)
python stage3_dp.py         # input-DP and output-DP experiments
python stage4_evaluation.py # re-ID attack, reconstruction attack, figures
```

All paths are configured in `config.py`. Outputs go to `outputs/`.

## Key Findings

1. **Input perturbation DP is neutralised by NLDR** — noise added before compression is projected away (PCA) or absorbed by the k-NN graph (UMAP/t-SNE), leaving downstream accuracy unchanged.
2. **Non-linear methods provide inherent reconstruction privacy** — a linear decoder recovers far less from a UMAP/t-SNE embedding than from a PCA embedding, even without formal DP.
