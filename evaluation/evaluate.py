from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.epoch_dataset import get_record_paths
from data.sequence_dataset import SequenceDataset
from evaluation.metrics import save_metrics
from training.common import load_config, seed_everything
from training.finetune import _torch_load, build_model_from_finetuned_checkpoint, evaluate_sequence_model


def run_from_config(path: str | Path):
    cfg = load_config(path)
    seed_everything(int(cfg.get("seed", 777)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _torch_load(cfg["checkpoint"])
    if ckpt.get("channel_mode") and ckpt["channel_mode"] != cfg["channel_mode"]:
        raise ValueError("Evaluation channel mode does not match checkpoint")
    model = build_model_from_finetuned_checkpoint(ckpt).to(device)
    paths = get_record_paths(cfg["manifest"], cfg["split"], cfg.get("partition", "test"), cfg["channel_mode"])
    s = cfg["sequence"]
    ds = SequenceDataset(paths, context_length=int(s["context_length"]), stride=int(s["stride"]))
    loader = DataLoader(ds, batch_size=int(s["batch_size"]), shuffle=False, num_workers=int(s.get("num_workers", 0)))
    _, summary, y_true, y_pred = evaluate_sequence_model(model, loader, device, criterion=None)
    save_metrics(cfg["output_dir"], "evaluation", y_true, y_pred)
    print(summary)
    return summary
