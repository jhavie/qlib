# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
FMZ Volume Single-Factor DataHandler for crypto perpetual futures.

Two-stage computation (same architecture as handler_alpha101_fmz.py):
  Stage 1 — qlib expression engine reads $amount (= Binance quote_volume / USDT).
  Stage 2 — VolumeFMZComposite Processor applies cross-sectional quantile clip + z-score.

IMPORTANT field mapping:
  FMZ "volume" = Binance col[7] = quote_volume (USDT turnover)
  qlib $volume = Binance col[5] = base_qty (coin quantity)
  qlib $amount = Binance col[7] = quote_volume (USDT turnover)  ← USE THIS

Usage in YAML:
    handler:
      class: VolumeFMZHandler
      module_path: qlib.contrib.data.handler_volume_fmz
      kwargs:
        instruments: all
        start_time: "2022-01-01"
        end_time: "2022-12-31"
        freq: "240min"
"""

import numpy as np
import pandas as pd

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.contrib.data.handler import check_transform_proc


# ---------------------------------------------------------------------------
# Stage 1: Time-series feature expressions (per-instrument, qlib engine)
# ---------------------------------------------------------------------------
# $amount = quote_volume (USDT turnover), which is FMZ's "volume"
_TS_FEATURE_EXPRS = ["$amount"]
_TS_FEATURE_NAMES = ["VOLUME"]


# ---------------------------------------------------------------------------
# Stage 2: Cross-Sectional Processor
# ---------------------------------------------------------------------------
class VolumeFMZComposite(Processor):
    """
    Reproduces FMZ's norm_factor exactly:
      1. Cross-sectional quantile(0.2, 0.8) clip per bar
      2. Cross-sectional z-score
      3. Negate (FMZ: long=low values → qlib: buy=high scores)
    """

    def __init__(self, negate_composite=True, **kwargs):
        self.negate_composite = negate_composite

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        volume = df["VOLUME"]

        # 1. Cross-sectional quantile(0.2, 0.8) clip
        q20 = volume.groupby("datetime").transform(lambda x: x.quantile(0.2))
        q80 = volume.groupby("datetime").transform(lambda x: x.quantile(0.8))
        clipped = volume.clip(lower=q20, upper=q80)

        # 2. Cross-sectional z-score
        mu = clipped.groupby("datetime").transform("mean")
        sigma = clipped.groupby("datetime").transform("std")
        sigma = sigma.replace(0, np.nan)
        z = ((clipped - mu) / sigma).fillna(0)

        # 3. Negate for qlib convention (high composite = buy candidate)
        if self.negate_composite:
            z = -z

        df["COMPOSITE"] = z

        # Keep only COMPOSITE
        result = df[["COMPOSITE"]]
        result.columns = pd.MultiIndex.from_tuples([("feature", "COMPOSITE")])
        return result

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# DataHandler
# ---------------------------------------------------------------------------
class VolumeFMZHandler(DataHandlerLP):
    """
    FMZ Volume single-factor DataHandler for crypto perpetual futures.

    YAML-configurable parameters:
        instruments, start_time, end_time, freq, negate_composite
    """

    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        freq="240min",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        negate_composite=True,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = [
                {
                    "class": "VolumeFMZComposite",
                    "module_path": "qlib.contrib.data.handler_volume_fmz",
                    "kwargs": {
                        "negate_composite": negate_composite,
                    },
                },
            ]

        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
            ]

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=DataHandlerLP.PTYPE_I,
            **kwargs,
        )

    @staticmethod
    def get_feature_config():
        return [_TS_FEATURE_EXPRS, _TS_FEATURE_NAMES]

    @staticmethod
    def get_label_config():
        # 4h crypto: next-bar return
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
