# Sleep Stage Classification From Multichannel Polysomnography Data Using a NeuroNet Model

> **Phạm Thị Anh Đào, Dương Thị Quỳnh Giang**

[[`Original NeuroNet`](https://github.com/dlcjfgmlnasa/NeuroNet)]
[[`NeuroNet Paper`](https://arxiv.org/abs/2404.17585)]
[[`BibTeX`](#citation)]

## Introduction 🔥

Automatic sleep stage classification is an important task in sleep analysis, but manual scoring of polysomnography (PSG) recordings is time-consuming and depends on expert annotation. NeuroNet introduced a hybrid self-supervised learning framework for sleep staging from **single-channel EEG**, combining masked prediction and contrastive learning.

This work extends NeuroNet from single-channel EEG to a **multichannel EEG + EOG setting**. The main goal is to study whether complementary eye-movement information from EOG improves five-stage sleep classification, especially for difficult stages such as **N1** and **REM**.

We evaluate two input settings under the same experimental pipeline:

- **Single-channel:** EEG only.
- **Multichannel:** EEG + EOG.

HMC and Sleep-EDF are used as source-domain datasets for preprocessing, self-supervised pretraining, supervised fine-tuning, and internal evaluation. ISRUC is used to assess cross-dataset generalization and optional target-domain adaptation.

## Highlights ✨

- Extends NeuroNet from **single-channel EEG** to **EEG + EOG**.
- Supports unified preprocessing from raw PSG/EDF recordings to NPZ.
- Uses a shared per-channel ResNet1D backbone followed by feature fusion for multichannel input.
- Uses participant-wise train/validation/test splitting to reduce data leakage.
- Keeps the test set untouched during checkpoint selection.
- Uses the NeuroNet encoder representation with a **TCM-LSTM** for supervised sequence classification.
- Evaluates both **internal performance** on HMC + Sleep-EDF and **cross-dataset generalization** on ISRUC.

## Overview

```text
Raw PSG / EDF
      │
      ▼
Signal preprocessing
(filter 1–50 Hz, resample 100 Hz, 30 s epochs, z-score)
      │
      ▼
NPZ dataset
      │
      ├───────────────┐
      │               │
      ▼               ▼
Single-channel     Multichannel
EEG                EEG + EOG
      │               │
      └───────┬───────┘
              ▼
Self-supervised NeuroNet pretraining
Masked Prediction + Contrastive Learning
              │
              ▼
Pretrained NeuroNet encoder
              │
              ▼
20-epoch temporal sequences
              │
              ▼
TCM-LSTM + classification head
              │
              ▼
Wake / N1 / N2 / N3 / REM
              │
              ├── Internal test: HMC + Sleep-EDF
              └── External test / adaptation: ISRUC
```

## Datasets

| Dataset | Role | Single-channel | Multichannel |
|---|---|---|---|
| **HMC** | Source-domain training and internal evaluation | EEG C4-M1 | EEG C4-M1 + EOG E1-M2 |
| **Sleep-EDF** | Source-domain training and internal evaluation | EEG Fpz-Cz | EEG Fpz-Cz + EOG horizontal |
| **ISRUC** | Cross-dataset evaluation / target-domain adaptation | EEG selected by configured priority | EEG + EOG selected by configured priority |

All signals are converted to 30-second epochs at 100 Hz. The classification target contains five sleep stages:

```text
0: Wake
1: N1
2: N2
3: N3
4: REM
```

## Model

The implementation inherits the main NeuroNet components:

- multiscale **ResNet1D** frame backbone;
- Transformer encoder/decoder;
- masked prediction;
- NT-Xent contrastive objective.

For multichannel input, each signal channel is processed by a shared frame backbone and the resulting channel features are fused before the Transformer encoder.

During supervised fine-tuning, the encoder representation of consecutive epochs is passed to a temporal context module:

```text
NeuroNet encoder
      │
      ▼
Epoch representation
      │
      ▼
Embedding layer
      │
      ▼
2-layer LSTM
(hidden size = 256)
      │
      ▼
5-class classifier
```

The default temporal context contains **20 consecutive epochs**, corresponding to 10 minutes of PSG data.

## Repository Structure

```text
configs/        experiment and preprocessing configurations
preprocessing/  HMC, Sleep-EDF and ISRUC preprocessing
data/           manifests, participant splits and datasets
models/         NeuroNet, multichannel fusion and TCM
training/       self-supervised pretraining and supervised fine-tuning
evaluation/     metrics and evaluation utilities
scripts/        command-line entry points
notebooks/      result analysis only
datasets/       raw/processed data, manifests and splits
outputs/        checkpoints, metrics and logs
docs/           experiment protocol and upstream notes
```

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the raw datasets under:

```text
datasets/raw/hmc/
datasets/raw/sleepedf/
datasets/raw/isruc/
```

Edit `configs/preprocessing.yaml` when the paths or preprocessing policy need to be changed.

A small smoke test is recommended before processing the full datasets:

```bash
python scripts/preprocess.py --config configs/preprocessing.yaml --dataset hmc --limit 2
python scripts/preprocess.py --config configs/preprocessing.yaml --dataset sleepedf --limit 2
python scripts/preprocess.py --config configs/preprocessing.yaml --dataset isruc --limit 2
```

Process all datasets:

```bash
python scripts/preprocess.py --config configs/preprocessing.yaml --dataset all
```

## Build Manifests and Splits

Build the HMC + Sleep-EDF source-domain manifest:

```bash
python scripts/build_manifest.py \
  --processed-roots datasets/processed/hmc datasets/processed/sleepedf \
  --output datasets/manifests/source_manifest.json
```

Build the ISRUC manifest:

```bash
python scripts/build_manifest.py \
  --processed-roots datasets/processed/isruc \
  --output datasets/manifests/isruc_manifest.json
```

Create participant-wise splits:

```bash
python scripts/make_split.py --config configs/split_source.yaml
python scripts/make_split.py --config configs/split_isruc.yaml
```

The source-domain protocol uses a **60/20/20 participant-wise split** for train, validation, and test.

## Self-Supervised Pretraining

Single-channel:

```bash
python scripts/pretrain.py --config configs/pretrain_single.yaml
```

Multichannel:

```bash
python scripts/pretrain.py --config configs/pretrain_multi.yaml
```

Checkpoint selection follows:

```text
source train
   │
   ▼
SSL pretraining
   │
   ▼
probe fitted on train embeddings
   │
   ▼
validation Macro-F1
   │
   ▼
best checkpoint
   │
   ▼
test evaluation once
```

## Supervised Fine-Tuning

Single-channel:

```bash
python scripts/finetune.py --config configs/finetune_single.yaml
```

Multichannel:

```bash
python scripts/finetune.py --config configs/finetune_multi.yaml
```

The default fine-tuning setup uses:

- context length: 20 epochs;
- TCM: 2-layer LSTM;
- hidden size: 256;
- classification loss: CrossEntropyLoss;
- checkpoint selection: validation Macro-F1.

## ISRUC Cross-Dataset Evaluation

Evaluate source-trained models directly on ISRUC:

```bash
python scripts/evaluate.py --config configs/isruc_zero_shot_single.yaml
python scripts/evaluate.py --config configs/isruc_zero_shot_multi.yaml
```

Optional target-domain adaptation:

```bash
python scripts/finetune.py --config configs/isruc_adapt_single.yaml
python scripts/finetune.py --config configs/isruc_adapt_multi.yaml
```

During adaptation, ISRUC train and validation are used for optimization and checkpoint selection, while the ISRUC test set is reserved for final evaluation.

## Evaluation Metrics

The main metrics are:

- Accuracy;
- Macro Precision;
- Macro Recall;
- Macro-F1;
- Weighted-F1;
- per-class Precision / Recall / F1-score;
- confusion matrix.

## Main Results 🥇

## Main Results 🥇

### Single-channel vs. Multichannel

The following results compare the single-channel setting (EEG) and the
multichannel setting (EEG + EOG) on the HMC + Sleep-EDF source-domain
experiments.

| Metric | Single Pretraining | Single Fine-tuning | Multi Pretraining | Multi Fine-tuning |
|---|---:|---:|---:|---:|
| Accuracy | 64.39 | 76.99 | 65.08 | **78.61** |
| Macro Precision | 58.20 | 73.71 | 59.20 | **75.03** |
| Macro Recall | 57.19 | 72.67 | 58.72 | **75.28** |
| Macro-F1 | 57.34 | 73.10 | 58.54 | **74.93** |
| Weighted-F1 | 63.18 | 76.87 | 63.89 | **78.36** |

After supervised fine-tuning, the multichannel model outperformed the
single-channel model across all overall metrics. Macro-F1 increased from
**73.10% to 74.93%**, while Accuracy increased from **76.99% to 78.61%**.
The largest improvement among the overall metrics was observed in Macro Recall,
which increased from **72.67% to 75.28%**.


## Acknowledgement

This repository is a derived implementation of the original **NeuroNet** project:

> Cheol-Hui Lee, Hakseung Kim, Hyun-jee Han, Min-Kyung Jung, Byung C. Yoon, and Dong-Joo Kim,  
> *NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG.*

Original repository:  
https://github.com/dlcjfgmlnasa/NeuroNet

Original paper:  
https://arxiv.org/abs/2404.17585

Please retain the original NeuroNet license and citation when redistributing derived code.

## Citation

A paper citation will be added after publication. For the current research version:

```bibtex
@misc{dao2026multichannelneuronet,
  title  = {Sleep Stage Classification From Multichannel Polysomnography Data Using a NeuroNet Model},
  author = {Pham Thi Anh Dao and Duong Thi Quynh Giang},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

If you use this implementation, please also cite the original NeuroNet paper:

```bibtex
@misc{lee2024neuronet,
  title        = {NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG},
  author       = {Cheol-Hui Lee and Hakseung Kim and Hyun-jee Han and Min-Kyung Jung and Byung C. Yoon and Dong-Joo Kim},
  year         = {2024},
  eprint       = {2404.17585},
  archivePrefix= {arXiv},
  primaryClass = {cs.HC}
}
```

## License

This project is derived from NeuroNet. Keep the upstream NeuroNet `LICENSE` file in the repository and ensure that redistributed code complies with the original license terms.
