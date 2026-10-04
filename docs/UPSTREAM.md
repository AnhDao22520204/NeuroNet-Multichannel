# Upstream NeuroNet

This repository is intended to be maintained as a fork/derived implementation of:

- Repository: `https://github.com/dlcjfgmlnasa/NeuroNet`
- Core inherited components: multiscale ResNet1D frame backbone, Transformer
  encoder/decoder, masked prediction, and NT-Xent contrastive objective.

Before publishing the paper repository, record the exact upstream commit hash here
and retain the upstream LICENSE from the original repository.

## Thesis/paper changes consolidated in this source

1. EDF/PSG -> NPZ preprocessing for HMC, Sleep-EDF and ISRUC.
2. Single-channel EEG and multichannel EEG+EOG data paths.
3. Shared per-channel ResNet1D followed by mean feature fusion for multichannel input.
4. One participant-wise split shared by single and multichannel experiments.
5. Pretraining checkpoint selection on validation only; test is evaluated once.
6. Supervised TCM-LSTM fine-tuning using the CLS token.
7. Evaluation aggregates overlapping sequence predictions by epoch so each real
   epoch is counted once in validation/test metrics.
8. ISRUC label mapping is deterministic and configured explicitly.
