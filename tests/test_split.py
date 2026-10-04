import json
from pathlib import Path

from data.split import make_subject_split


def test_sleepedf_nights_stay_in_same_subject_split(tmp_path):
    manifest = {
        "schema_version": 1,
        "records": [],
    }
    # Two nights for each SleepEDF participant + enough HMC participants for stratification.
    for sid in ["SC4001", "SC4011", "SC4021", "SC4031", "SC4041"]:
        for night in ["E0", "E1"]:
            manifest["records"].append({
                "dataset": "SleepEDF", "subject_id": sid, "record_id": sid + night,
                "single_path": "dummy", "multi_path": "dummy", "n_epochs": 1,
            })
    for i in range(5):
        sid = f"SN{i:03d}"
        manifest["records"].append({
            "dataset": "HMC", "subject_id": sid, "record_id": sid,
            "single_path": "dummy", "multi_path": "dummy", "n_epochs": 1,
        })
    mp = tmp_path / "manifest.json"
    sp = tmp_path / "split.json"
    mp.write_text(json.dumps(manifest))
    split = make_subject_split(mp, sp, datasets=["HMC", "SleepEDF"], seed=42)
    sets = {name: set(split[name]) for name in ["train", "val", "test"]}
    for sid in ["SC4001", "SC4011", "SC4021", "SC4031", "SC4041"]:
        key = f"SleepEDF::{sid}"
        assert sum(key in s for s in sets.values()) == 1
