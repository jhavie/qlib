# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
FMZ Alpha044 Single-Factor DataHandler for crypto perpetual futures.

Alpha044 = corr(high, cs_rank(volume), 6)

Two-stage computation:
  Stage 1 — qlib expression engine reads $high and $volume (= base_qty).
  Stage 2 — Alpha044FMZComposite Processor computes:
      1. Cross-sectional rank of volume per bar
      2. Per-instrument rolling correlation(high, cs_rank_vol, window)
      Output raw correlation values — no z-score, no negate.

IMPORTANT field mapping:
  qlib $volume = Binance col[5] = base_qty (coin quantity) ← USE THIS
  qlib $amount = Binance col[7] = quote_volume (USDT turnover)

Usage in YAML:
    handler:
      class: Alpha044FMZHandler
      module_path: qlib.contrib.data.handler_alpha044_fmz
      kwargs:
        instruments: all
        start_time: "2022-01-01"
        end_time: "2022-12-31"
        freq: "240min"
        window: 6
"""

import pandas as pd

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.contrib.data.handler import check_transform_proc


# ---------------------------------------------------------------------------
# Stage 1: Time-series feature expressions (per-instrument, qlib engine)
# ---------------------------------------------------------------------------
_TS_FEATURE_EXPRS = ["$high", "$volume"]
_TS_FEATURE_NAMES = ["HIGH", "VOLUME"]


# ---------------------------------------------------------------------------
# Stage 2: Cross-Sectional + Rolling Processor
# ---------------------------------------------------------------------------
class Alpha044FMZComposite(Processor):
    """
    Computes Alpha044 = corr(high, cs_rank(volume), window).

    Steps:
      1. Cross-sectional rank of VOLUME per bar (datetime).
      2. Per-instrument rolling Pearson correlation between HIGH and cs_rank_vol.
      3. Output raw correlation — no z-score, no negate.
    """

    def __init__(self, window=5, **kwargs):
        self.window = window

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        high = df["HIGH"]
        volume = df["VOLUME"]

        # 1. Cross-sectional rank of volume per bar (percent rank 0~1)
        #    method='min' aligns with nautilus_quants popbo convention:
        #    rank = (count_strictly_less + 1) / n, values in [1/n, 1]
        cs_rank_vol = volume.groupby("datetime").rank(method="min", pct=True)

        # 2. Per-instrument rolling correlation(high, cs_rank_vol, window)
        # Build a temporary DataFrame aligned on the same index
        tmp = pd.DataFrame({"high": high, "cs_rank_vol": cs_rank_vol})
        corr = (
            tmp.groupby("instrument")
            .apply(
                lambda g: g["high"].rolling(self.window, min_periods=self.window).corr(g["cs_rank_vol"]),
            )
        )
        # apply() prepends group key → 3-level index; drop outer level and realign
        corr = corr.droplevel(0).reindex(df.index)

        df["COMPOSITE"] = -corr

        # Keep only COMPOSITE
        result = df[["COMPOSITE"]].copy()
        result.columns = pd.MultiIndex.from_tuples([("feature", "COMPOSITE")])
        return result

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# DataHandler
# ---------------------------------------------------------------------------
class Alpha044FMZHandler(DataHandlerLP):
    """
    FMZ Alpha044 single-factor DataHandler for crypto perpetual futures.

    Alpha044 = corr(high, cs_rank(volume), window)

    YAML-configurable parameters:
        instruments, start_time, end_time, freq, window
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
        window=6,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = [
                {
                    "class": "Alpha044FMZComposite",
                    "module_path": "qlib.contrib.data.handler_alpha044_fmz",
                    "kwargs": {
                        "window": window,
                    },
                },
            ]

        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
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
