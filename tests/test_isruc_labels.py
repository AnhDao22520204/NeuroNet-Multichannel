from preprocessing.isruc import _parse_scalar


def test_rk_0_5_mapping_is_deterministic():
    assert _parse_scalar("0", "rk_0_5") == 0
    assert _parse_scalar("3", "rk_0_5") == 3
    assert _parse_scalar("4", "rk_0_5") == 3  # N4 -> N3
    assert _parse_scalar("5", "rk_0_5") == 4  # REM
