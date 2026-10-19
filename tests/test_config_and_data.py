import copy

import pandas as pd
import pytest

from jaxon.config import ConfigError, from_dict, load_config, validate
from jaxon.data.csv_source import DataValidationError, load_instrument_csv
from tests.helpers import ROOT, base_cfg_dict, cfg


# ---------------------------------------------------------------- config
@pytest.mark.parametrize("name", ["example_equity.yaml", "example_crypto.yaml", "example_scenario.yaml",
                                  "templates/futures_template.yaml", "templates/forex_template.yaml"])
def test_example_configs_validate(name):
    c = load_config(ROOT / "configs" / name)
    assert c.example is True
    assert c.config_hash and c.version_tag.startswith(c.config_version)


@pytest.mark.parametrize("field", ["tick_size", "lot_size", "value_per_point", "min_quantity", "asset_class",
                                   "volume_type", "session", "costs", "currency"])
def test_missing_mandatory_instrument_metadata_rejected(field):
    d = base_cfg_dict()
    del d["instruments"][0][field]
    with pytest.raises(ConfigError, match=field):
        from_dict(d)


def test_unknown_key_and_bad_values_rejected():
    d = base_cfg_dict()
    d["instruments"][0]["tick_sise"] = 0.01
    with pytest.raises(ConfigError, match="unknown keys"):
        from_dict(d)
    d = base_cfg_dict(tick_size=0)
    with pytest.raises(ConfigError, match="tick_size"):
        from_dict(d)
    d = base_cfg_dict(currency="EUR")
    with pytest.raises(ConfigError, match="FX conversion"):
        from_dict(d)
    d = base_cfg_dict()
    d["instruments"][0]["bars"]["base_interval_minutes"] = 3
    with pytest.raises(ConfigError, match="divide 5"):
        from_dict(d)


def test_tick_volume_warns_vwap_disabled():
    c = cfg(volume_type="tick", asset_class="forex")
    assert any("not allowed for VWAP" in w for w in validate(c))


def test_config_hash_changes_with_parameters():
    a = cfg()
    b = copy.deepcopy(a)
    b.strategy.min_reward_risk = 2.0
    assert a.config_hash != b.config_hash


# ---------------------------------------------------------------- CSV import
def _write(tmp_path, rows, header="timestamp,open,high,low,close,volume", comment=None):
    p = tmp_path / "d.csv"
    lines = ([f"# {comment}"] if comment else []) + [header] + rows
    p.write_text("\n".join(lines) + "\n")
    return p


def _inst(p, **bars_over):
    c = cfg(path=str(p))
    for k, v in bars_over.items():
        setattr(c.instruments[0].bars, k, v)
    return c.instruments[0]


def test_timestamps_normalized_to_utc(tmp_path):
    p = _write(tmp_path, ["2025-03-07 09:30,10,11,9,10,100", "2025-03-07 09:35,10,11,9,10,100"])
    inst = _inst(p, naive_timestamp_timezone="America/New_York")
    bars, prov, rep = load_instrument_csv(inst)
    assert bars[0].start == pd.Timestamp("2025-03-07 14:30", tz="UTC")
    inst2 = _inst(p)
    with pytest.raises(DataValidationError, match="no UTC offset"):
        load_instrument_csv(inst2)


def test_offset_and_epoch_timestamps(tmp_path):
    p = _write(tmp_path, ["2025-03-07T09:30:00-05:00,10,11,9,10,1"])
    assert load_instrument_csv(_inst(p))[0][0].start == pd.Timestamp("2025-03-07 14:30", tz="UTC")
    p = _write(tmp_path, ["1741357800000,10,11,9,10,1"])
    assert load_instrument_csv(_inst(p))[0][0].start == pd.Timestamp("2025-03-07 14:30", tz="UTC")


def test_close_convention_shifts_to_open(tmp_path):
    p = _write(tmp_path, ["2025-03-07T14:35:00Z,10,11,9,10,1"])
    b = load_instrument_csv(_inst(p, timestamp_convention="close"))[0][0]
    assert b.start == pd.Timestamp("2025-03-07 14:30", tz="UTC")


def test_ordering_and_duplicates(tmp_path):
    rows = ["2025-01-01T00:05:00Z,10,11,9,10,1", "2025-01-01T00:00:00Z,10,11,9,10,1",
            "2025-01-01T00:00:00Z,10,11,9,10,1"]
    bars, _, rep = load_instrument_csv(_inst(_write(tmp_path, rows)))
    assert [b.start.minute for b in bars] == [0, 5]
    assert rep.out_of_order_rows == 1 and rep.exact_duplicates_dropped == 1
    rows.append("2025-01-01T00:00:00Z,10,12,9,10,1")
    with pytest.raises(DataValidationError, match="conflicting duplicate"):
        load_instrument_csv(_inst(_write(tmp_path, rows)))


@pytest.mark.parametrize("row,msg", [
    ("2025-01-01T00:00:00Z,10,9,8,10,1", "OHLC"),       # high < close
    ("2025-01-01T00:00:00Z,10,11,10.5,10,1", "OHLC"),   # low > close
    ("2025-01-01T00:00:00Z,0,0,0,0,1", "OHLC"),         # non-positive
    ("2025-01-01T00:00:00Z,10,11,9,,1", "non-numeric"),
    ("2025-01-01T00:02:00Z,10,11,9,10,1", "aligned"),
    ("2025-01-01T00:00:00Z,10,11,9,10,-5", "negative volume"),
])
def test_invalid_rows_rejected(tmp_path, row, msg):
    with pytest.raises(DataValidationError, match=msg):
        load_instrument_csv(_inst(_write(tmp_path, [row])))


def test_missing_columns_rejected(tmp_path):
    p = _write(tmp_path, ["2025-01-01T00:00:00Z,10,11,9,10"], header="timestamp,open,high,low,close")
    with pytest.raises(DataValidationError, match="missing columns"):
        load_instrument_csv(_inst(p))


def test_gaps_reported_and_never_filled(tmp_path):
    rows = ["2025-01-01T00:00:00Z,10,11,9,10,1", "2025-01-01T00:20:00Z,10,11,9,10,1"]
    bars, _, rep = load_instrument_csv(_inst(_write(tmp_path, rows)))
    assert len(bars) == 2                          # nothing manufactured
    assert rep.gaps[0]["missing_bars"] == 3


def test_missing_volume_allowed_but_flagged(tmp_path):
    bars, _, rep = load_instrument_csv(_inst(_write(tmp_path, ["2025-01-01T00:00:00Z,10,11,9,10,"])))
    assert rep.missing_volume_rows == 1 and not bars[0].volume_ok


def test_outside_session_rows_dropped(tmp_path):
    c = load_config(ROOT / "configs" / "example_equity.yaml")
    inst = c.instruments[0]
    inst.data.path = str(_write(tmp_path, ["2025-03-07T14:00:00Z,10,11,9,10,1",       # 09:00 ET pre-market
                                           "2025-03-07T14:30:00Z,10,11,9,10,1"]))
    bars, _, rep = load_instrument_csv(inst)
    assert len(bars) == 1 and rep.outside_session_rows_dropped == 1


def test_synthetic_header_detected(tmp_path):
    p = _write(tmp_path, ["2025-01-01T00:00:00Z,10,11,9,10,1"], comment="SYNTHETIC TEST FIXTURE")
    inst = _inst(p)
    inst.data.synthetic = False
    _, prov, _ = load_instrument_csv(inst)
    assert prov.synthetic is True
    p2 = _write(tmp_path, ["2025-01-01T00:00:00Z,10,11,9,10,1"])
    _, prov2, _ = load_instrument_csv(_inst(p2))  # declared synthetic in cfg()
    assert prov2.synthetic is True
    inst3 = _inst(p2)
    inst3.data.synthetic = False
    assert load_instrument_csv(inst3)[1].synthetic is False
