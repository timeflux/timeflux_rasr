import numpy as np
import pandas as pd
import pytest

from timeflux_rasr.helpers.config import Config
from timeflux_rasr.helpers.utils import (
    float_index_to_time_index,
    time_index_to_float_index,
)


def test_float_index_to_time_index_round_trip():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0]}, index=[1.5e9, 1.5e9 + 0.004, 1.5e9 + 0.008]
    )
    dt = float_index_to_time_index(df)
    assert dt.index.dtype == "datetime64[us]"
    assert dt.index[0] == pd.Timestamp("2017-07-14 02:40:00")
    np.testing.assert_allclose(time_index_to_float_index(dt).index, df.index)


def test_float_index_to_time_index_nan():
    # issue #16: pyxdf can produce NaN timestamps
    df = pd.DataFrame({"a": [1.0, 2.0]}, index=[1.5e9, np.nan])
    dt = float_index_to_time_index(df)
    assert pd.isna(dt.index[1])
    assert np.isnan(time_index_to_float_index(dt).index[1])


def test_config_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("TIMEFLUX_RASR_DATA", raising=False)
    with pytest.raises(ValueError, match="TIMEFLUX_RASR_DATA"):
        Config()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "s1.xdf").touch()
    monkeypatch.setenv("TIMEFLUX_RASR_DATA", str(tmp_path))
    config = Config()
    assert config.raw_files == [str(tmp_path / "raw" / "s1.xdf")]
    assert (tmp_path / "results").is_dir()
