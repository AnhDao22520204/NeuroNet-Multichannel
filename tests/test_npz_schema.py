import numpy as np
from preprocessing.common import save_npz
from data.schema import inspect_npz


def test_clean_npz_schema(tmp_path):
    path = tmp_path / "x.npz"
    x = np.zeros((3, 2, 3000), dtype=np.float32)
    y = np.array([0, 2, 4], dtype=np.int64)
    save_npz(
        path, x=x, y=y, fs=100, channel_roles=["EEG", "EOG"],
        original_channel_names=["C4-M1", "E1-M2"], subject_id="SN001",
        record_id="SN001", dataset="HMC", preprocessing_config={"test": True},
    )
    meta = inspect_npz(path)
    assert meta["n_channels"] == 2
    assert meta["n_samples"] == 3000
    assert meta["labels"] == [0, 2, 4]
