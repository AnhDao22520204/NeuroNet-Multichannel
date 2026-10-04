from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.epoch_dataset import get_record_paths
from data.sequence_dataset import SequenceDataset
from evaluation.metrics import compute_metrics, save_metrics
from evaluation.sequence import aggregate_sequence_predictions
from models.neuronet.model import NeuroNet, NeuroNetEncoderWrapper
from models.neuronet.tcm import build_tcm
from training.common import cpu_state_dict, load_config, seed_everything


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model_from_pretrain_checkpoint(checkpoint: dict, finetune_cfg: dict):
    params = checkpoint["model_parameter"]
    base = NeuroNet(**params)
    base.load_state_dict(checkpoint["model_state"])
    encoder = NeuroNetEncoderWrapper.from_neuronet(base, pooling="cls")
    train_cfg = finetune_cfg["training"]
    return build_tcm(
        train_cfg.get("temporal_context_module", "lstm"),
        encoder,
        base.autoencoder.embed_dim,
        int(train_cfg.get("embed_dim", 256)),
        int(train_cfg.get("unfreeze_last_encoder_blocks", 1)),
    ), params


def build_model_from_finetuned_checkpoint(checkpoint: dict):
    params = checkpoint["pretrain_model_parameter"]
    base = NeuroNet(**params)
    encoder = NeuroNetEncoderWrapper.from_neuronet(base, pooling="cls")
    model_cfg = checkpoint["model_config"]
    model = build_tcm(
        model_cfg["temporal_context_module"],
        encoder,
        base.autoencoder.embed_dim,
        int(model_cfg["embed_dim"]),
        int(model_cfg.get("unfreeze_last_encoder_blocks", 1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model



@torch.no_grad()
def evaluate_sequence_model(model, loader, device, criterion=None):
    model.eval()
    losses = []
    logits_batches, labels_batches, record_batches, index_batches = [], [], [], []
    for x, y, record_idx, epoch_idx in loader:
        x = x.to(device, non_blocking=True)
        y_device = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            losses.append(float(criterion(logits.reshape(-1, 5), y_device.reshape(-1)).cpu()))
        logits_batches.append(logits.cpu().numpy())
        labels_batches.append(y.numpy())
        record_batches.append(record_idx.numpy())
        index_batches.append(epoch_idx.numpy())
    y_true, y_pred = aggregate_sequence_predictions(logits_batches, labels_batches, record_batches, index_batches)
    summary, _, _ = compute_metrics(y_true, y_pred)
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return avg_loss, summary, y_true, y_pred


class FineTuneRunner:
    def __init__(self, config: dict):
        self.cfg = config
        exp = config["experiment"]
        self.seed = int(exp["seed"])
        seed_everything(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(exp["output_dir"])
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.channel_mode = exp["channel_mode"]
        self.manifest = exp["manifest"]
        self.split = exp["split"]
        self.train_paths = get_record_paths(self.manifest, self.split, "train", self.channel_mode)
        self.val_paths = get_record_paths(self.manifest, self.split, "val", self.channel_mode)
        self.test_paths = get_record_paths(self.manifest, self.split, "test", self.channel_mode)

        initial_ft = exp.get("initial_finetuned_checkpoint")
        if initial_ft:
            initial_ckpt = _torch_load(initial_ft)
            self.model = build_model_from_finetuned_checkpoint(initial_ckpt)
            self.pretrain_model_parameter = initial_ckpt["pretrain_model_parameter"]
            # Re-apply requested freeze policy to the loaded model.
            self._reset_freeze_policy(int(config["training"].get("unfreeze_last_encoder_blocks", 1)))
        else:
            pretrain_path = exp.get("pretrain_checkpoint")
            if not pretrain_path:
                raise ValueError("pretrain_checkpoint is required when initial_finetuned_checkpoint is null")
            pretrain_ckpt = _torch_load(pretrain_path)
            if pretrain_ckpt.get("channel_mode") and pretrain_ckpt["channel_mode"] != self.channel_mode:
                raise ValueError("Channel mode does not match pretrain checkpoint")
            self.model, self.pretrain_model_parameter = build_model_from_pretrain_checkpoint(pretrain_ckpt, config)
        self.model = self.model.to(self.device)

        t = config["training"]
        self.epochs = int(t["epochs"])
        self.batch_size = int(t["batch_size"])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(t["learning_rate"]))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    def _reset_freeze_policy(self, n_last_blocks: int):
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return
        for p in backbone.parameters():
            p.requires_grad = False
        blocks = getattr(backbone, "encoder_block", None)
        if blocks is not None and n_last_blocks > 0:
            for block in list(blocks)[-n_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        norm = getattr(backbone, "encoder_norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True

    def _loader(self, paths, stride, shuffle):
        s = self.cfg["sequence"]
        t = self.cfg["training"]
        ds = SequenceDataset(paths, context_length=int(s["context_length"]), stride=int(stride))
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=int(t.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )

    def run(self):
        s = self.cfg["sequence"]
        train_loader = self._loader(self.train_paths, s["train_stride"], True)
        val_loader = self._loader(self.val_paths, s["eval_stride"], False)
        test_loader = self._loader(self.test_paths, s["eval_stride"], False)

        best_mf1 = -1.0
        best_state = None
        best_epoch = -1
        history = []

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for x, y, _, _ in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = self.criterion(logits.reshape(-1, 5), y.reshape(-1))
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_loss, val_summary, val_y, val_pred = evaluate_sequence_model(self.model, val_loader, self.device, self.criterion)
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_summary.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(self.metrics_dir / "validation_history.csv", index=False)

            if val_summary["macro_f1"] > best_mf1:
                best_mf1 = val_summary["macro_f1"]
                best_state = cpu_state_dict(self.model)
                best_epoch = epoch
                save_metrics(self.metrics_dir, "best_validation", val_y, val_pred, {"best_epoch": epoch + 1, "val_loss": val_loss})
                self._save_checkpoint(best_state, best_epoch, best_mf1)

            print(
                f"epoch={epoch+1:03d} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_summary['accuracy']:.4f} val_macro_f1={val_summary['macro_f1']:.4f}"
            )
            self.scheduler.step()

        if best_state is None:
            raise RuntimeError("No best fine-tuning state")
        self.model.load_state_dict(best_state)
        test_loss, test_summary, test_y, test_pred = evaluate_sequence_model(self.model, test_loader, self.device, self.criterion)
        save_metrics(self.metrics_dir, "final_test", test_y, test_pred, {"best_epoch": best_epoch + 1, "test_loss": test_loss})
        self._save_checkpoint(best_state, best_epoch, best_mf1, final_test_summary=test_summary)
        print(f"FINAL TEST: acc={test_summary['accuracy']:.4f} macro_f1={test_summary['macro_f1']:.4f}")
        return test_summary

    def _save_checkpoint(self, state, best_epoch, best_val_mf1, final_test_summary=None):
        t = self.cfg["training"]
        payload = {
            "model_name": "NeuroNet_TCM",
            "model_state": state,
            "channel_mode": self.channel_mode,
            "pretrain_model_parameter": self.pretrain_model_parameter,
            "model_config": {
                "embed_dim": int(t["embed_dim"]),
                "temporal_context_module": t["temporal_context_module"],
                "unfreeze_last_encoder_blocks": int(t.get("unfreeze_last_encoder_blocks", 1)),
            },
            "best_epoch": int(best_epoch + 1),
            "best_validation_macro_f1": float(best_val_mf1),
            "final_test_summary": final_test_summary,
            "config": self.cfg,
            "split_snapshot": {"manifest": self.manifest, "split": self.split},
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_dir / "best_model.pth")


def run_from_config(path: str | Path):
    return FineTuneRunner(load_config(path)).run()
