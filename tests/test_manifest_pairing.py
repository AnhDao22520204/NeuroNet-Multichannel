import numpy as np
from preprocessing.common import save_npz
from data.manifest import build_manifest


def test_manifest_requires_same_eeg_and_labels(tmp_path):
    root = tmp_path / "hmc"
    sdir = root / "single_channel"
    mdir = root / "multi_channel"
    sdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    eeg = np.arange(2 * 3000, dtype=np.float32).reshape(2, 1, 3000)
    y = np.array([0, 4])
    multi = np.concatenate([eeg, np.zeros_like(eeg)], axis=1)
    common = dict(fs=100, subject_id="SN001", record_id="SN001", dataset="HMC", preprocessing_config={})
    save_npz(sdir / "HMC__SN001.npz", x=eeg, y=y, channel_roles=["EEG"], original_channel_names=["C4-M1"], **common)
    save_npz(mdir / "HMC__SN001.npz", x=multi, y=y, channel_roles=["EEG","EOG"], original_channel_names=["C4-M1","E1-M2"], **common)
    out = tmp_path / "manifests" / "m.json"
    payload = build_manifest([root], out, strict_pairing=True)
    assert payload["summary"]["n_records"] == 1
