from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from data.epoch_dataset import EpochDataset, get_record_paths
from evaluation.metrics import compute_metrics, save_metrics
from models.neuronet.model import NeuroNet
from models.utils import model_size
from training.common import cpu_state_dict, load_config, seed_everything


def _subsample(x: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int):
    if not max_samples or len(y) <= max_samples:
        return x, y
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
        idx, _ = next(splitter.split(x, y))
    except Exception:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
    return x[idx], y[idx]


class PretrainRunner:
    def __init__(self, config: dict):
        self.cfg = config
        exp = config["experiment"]
        self.seed = int(exp["seed"])
        seed_everything(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(exp["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.channel_mode = exp["channel_mode"]
        self.manifest = exp["manifest"]
        self.split = exp["split"]

        self.train_paths = get_record_paths(self.manifest, self.split, "train", self.channel_mode)
        self.val_paths = get_record_paths(self.manifest, self.split, "val", self.channel_mode)
        self.test_paths = get_record_paths(self.manifest, self.split, "test", self.channel_mode)
        if not self.train_paths or not self.val_paths or not self.test_paths:
            raise RuntimeError("One of train/val/test path lists is empty")

        m = config["model"]
        self.model = NeuroNet(
            fs=int(m["fs"]),
            second=int(m["second"]),
            time_window=int(m["time_window"]),
            time_step=float(m["time_step"]),
            encoder_embed_dim=int(m["encoder_embed_dim"]),
            encoder_heads=int(m["encoder_heads"]),
            encoder_depths=int(m["encoder_depths"]),
            decoder_embed_dim=int(m["decoder_embed_dim"]),
            decoder_heads=int(m["decoder_heads"]),
            decoder_depths=int(m["decoder_depths"]),
            projection_hidden=list(m["projection_hidden"]),
            temperature=float(m["temperature"]),
        ).to(self.device)
        print(f"Model size: {model_size(self.model):.2f} MB")

        t = config["training"]
        self.epochs = int(t["epochs"])
        self.batch_size = int(config["data"]["batch_size"])
        self.accumulation = int(t.get("batch_accumulation", 1))
        effective_batch = self.batch_size * self.accumulation
        lr = float(t["base_learning_rate"]) * effective_batch / 256.0
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.mask_ratio = float(t["mask_ratio"])
        self.alpha = float(t["alpha"])
        self.print_every = int(t.get("print_every", 50))

    def _dataset(self, paths):
        d = self.cfg["data"]
        return EpochDataset(paths, scaler=bool(d.get("scaler", False)))

    def _loader(self, dataset, *, shuffle=False, batch_size=None):
        d = self.cfg["data"]
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=int(d.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )

    @torch.no_grad()
    def latent(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        xs, ys = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            z = self.model.forward_latent(x)
            xs.append(z.cpu().numpy())
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    def probe(self, train_loader, eval_loader, seed_offset=0):
        train_x, train_y = self.latent(train_loader)
        eval_x, eval_y = self.latent(eval_loader)
        p = self.cfg["probe"]
        train_x, train_y = _subsample(
            train_x,
            train_y,
            int(p.get("max_train_samples")) if p.get("max_train_samples") else None,
            self.seed + seed_offset,
        )
        n_comp = min(int(p.get("pca_components", 50)), train_x.shape[0], train_x.shape[1])
        pca = PCA(n_components=n_comp, random_state=self.seed)
        train_x = pca.fit_transform(train_x)
        eval_x = pca.transform(eval_x)
        knn = KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors", 5)))
        knn.fit(train_x, train_y)
        pred = knn.predict(eval_x)
        summary, _, _ = compute_metrics(eval_y, pred)
        return summary, eval_y, pred

    def run(self):
        train_dataset = self._dataset(self.train_paths)
        val_dataset = self._dataset(self.val_paths)
        train_loader = self._loader(train_dataset, shuffle=True)
        probe_batch = int(self.cfg["probe"].get("batch_size", 256))
        train_probe_loader = self._loader(train_dataset, shuffle=False, batch_size=probe_batch)
        val_loader = self._loader(val_dataset, shuffle=False, batch_size=probe_batch)

        best_mf1 = -1.0
        best_state = None
        best_epoch = -1
        rows = []
        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            running = []
            for step, (x, _) in enumerate(train_loader):
                x = x.to(self.device, non_blocking=True)
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = self.model(x, mask_ratio=self.mask_ratio)
                loss = recon_loss + self.alpha * contrastive_loss
                (loss / self.accumulation).backward()
                should_step = ((step + 1) % self.accumulation == 0) or (step + 1 == len(train_loader))
                if should_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                running.append(float(loss.detach().cpu()))
                if (global_step + 1) % self.print_every == 0:
                    cl_acc = (cl_logits.argmax(-1) == cl_labels).float().mean().item()
                    print(
                        f"epoch={epoch+1:03d} step={global_step+1:06d} "
                        f"recon={recon_loss.item():.4f} contrastive={contrastive_loss.item():.4f} "
                        f"total={loss.item():.4f} contrastive_acc={cl_acc:.4f}"
                    )
                global_step += 1

            # IMPORTANT: checkpoint selection uses TRAIN -> VALIDATION probe only.
            val_summary, val_y, val_pred = self.probe(train_probe_loader, val_loader, seed_offset=epoch)
            row = {
                "epoch": epoch + 1,
                "train_ssl_loss": float(np.mean(running)) if running else math.nan,
                **{f"val_{k}": v for k, v in val_summary.items()},
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(self.metrics_dir / "validation_history.csv", index=False)
            save_metrics(self.metrics_dir, f"validation_epoch_{epoch+1:03d}", val_y, val_pred, {"epoch": epoch + 1})

            if val_summary["macro_f1"] > best_mf1:
                best_mf1 = val_summary["macro_f1"]
                best_state = cpu_state_dict(self.model)
                best_epoch = epoch
                self._save_checkpoint(best_state, best_epoch, best_mf1)
            print(f"epoch={epoch+1:03d} val_acc={val_summary['accuracy']:.4f} val_macro_f1={val_summary['macro_f1']:.4f}")
            self.scheduler.step()

        if best_state is None:
            raise RuntimeError("No best checkpoint selected")
        self.model.load_state_dict(best_state)

        # TEST IS TOUCHED ONCE, after model selection is finished.
        test_dataset = self._dataset(self.test_paths)
        test_loader = self._loader(test_dataset, shuffle=False, batch_size=probe_batch)
        test_summary, test_y, test_pred = self.probe(train_probe_loader, test_loader, seed_offset=10000)
        save_metrics(self.metrics_dir, "final_test", test_y, test_pred, {"best_epoch": best_epoch + 1})
        print(f"FINAL TEST: acc={test_summary['accuracy']:.4f} macro_f1={test_summary['macro_f1']:.4f}")
        self._save_checkpoint(best_state, best_epoch, best_mf1, final_test_summary=test_summary)
        return test_summary

    def _save_checkpoint(self, state, best_epoch, best_val_mf1, final_test_summary=None):
        payload = {
            "model_name": "NeuroNet",
            "model_state": state,
            "model_parameter": dict(self.cfg["model"]),
            "channel_mode": self.channel_mode,
            "best_epoch": int(best_epoch + 1),
            "best_validation_macro_f1": float(best_val_mf1),
            "final_test_summary": final_test_summary,
            "config": self.cfg,
            "split_snapshot": {
                "manifest": self.manifest,
                "split": self.split,
                "train_paths": [str(p) for p in self.train_paths],
                "val_paths": [str(p) for p in self.val_paths],
                "test_paths": [str(p) for p in self.test_paths],
            },
        }
        torch.save(payload, self.output_dir / "best_model.pth")


def run_from_config(path: str | Path):
    return PretrainRunner(load_config(path)).run()
