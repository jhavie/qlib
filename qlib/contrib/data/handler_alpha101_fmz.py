# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
FMZ Alpha101 Multi-Factor DataHandler for crypto perpetual futures.

Two-stage computation:
  Stage 1 — qlib expression engine computes per-instrument time-series features.
  Stage 2 — Alpha101FMZComposite Processor executes cross-sectional ops + factor synthesis.

Usage in YAML:
    handler:
      class: Alpha101FMZHandler
      module_path: qlib.contrib.data.handler_alpha101_fmz
      kwargs:
        instruments: all
        start_time: "2022-01-01"
        end_time: "2025-12-31"
        freq: "4h"
        factor_weights:
          alpha044: 0.20
          alpha033: 0.15
          ...
"""

import numpy as np
import pandas as pd

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.contrib.data.handler import check_transform_proc


# ---------------------------------------------------------------------------
# Default factor weights (from FMZ backtest)
# ---------------------------------------------------------------------------
DEFAULT_FACTOR_WEIGHTS = {
    "alpha044": 0.20,
    "alpha033": 0.15,
    "alpha017": 0.15,
    "alpha026": 0.12,
    "alpha055": 0.12,
    "alpha002": 0.10,
    "alpha039": 0.10,
    "alpha003": 0.06,
}

# ---------------------------------------------------------------------------
# Stage 1: Time-series feature expressions (per-instrument, qlib engine)
# ---------------------------------------------------------------------------
# These are evaluated by QlibDataLoader; Rank() here is ts_rank (time-series percentile).
# Cross-sectional rank is done in Stage 2 Processor.

_TS_FEATURE_EXPRS = [
    # --- alpha044 inputs ---
    "$high",                    # HIGH
    "$volume",                  # VOLUME

    # --- alpha033 input ---
    "1 - $open / $close",      # A033_RAW

    # --- alpha017 inputs ---
    "Rank($close, 10)",        # A017_P1: ts_rank of close over 10 bars
    "$close - 2*Ref($close,1) + Ref($close,2)",  # A017_P2: delta of delta close
    "Rank($volume / Mean($volume, 20), 5)",       # A017_P3: ts_rank of vol ratio

    # --- alpha026 (fully time-series) ---
    "Max(Corr(Rank($volume,5), Rank($high,5), 5), 3)",  # A026

    # --- alpha055 input ---
    # RSV = (close - min_low_12) / (max_high_12 - min_low_12)
    "($close - Min($low, 12)) / If(Eq(Max($high, 12) - Min($low, 12), 0), 0.0001, Max($high, 12) - Min($low, 12))",  # A055_RSV

    # --- alpha002 inputs ---
    "Log($volume + 1e-12) - Ref(Log($volume + 1e-12), 2)",  # A002_LOGVD
    "($close - $open) / $open",  # A002_RET

    # --- alpha039 inputs ---
    "$close - Ref($close, 7)",    # A039_D7
    "WMA($volume / Mean($volume, 20), 9)",  # A039_WMA
    "Mean($close / Ref($close, 1) - 1, 250)",  # A039_MR

    # --- alpha003 input ---
    "$open",                     # OPEN
]

_TS_FEATURE_NAMES = [
    "HIGH",
    "VOLUME",
    "A033_RAW",
    "A017_P1",
    "A017_P2",
    "A017_P3",
    "A026",
    "A055_RSV",
    "A002_LOGVD",
    "A002_RET",
    "A039_D7",
    "A039_WMA",
    "A039_MR",
    "OPEN",
]


# ---------------------------------------------------------------------------
# Stage 2: Cross-Sectional Processor
# ---------------------------------------------------------------------------
class Alpha101FMZComposite(Processor):
    """
    Cross-sectional factor computation and weighted composite synthesis.

    Receives a MultiIndex (datetime, instrument) DataFrame from Stage 1 and:
    1. Computes 8 Alpha101 factors using cross-sectional rank + time-series corr.
    2. Z-score normalizes each factor cross-sectionally per bar.
    3. Produces a weighted composite signal column 'COMPOSITE'.

    The composite signal is NEGATED so that qlib's TopK strategy (which buys high
    scores) aligns with FMZ's convention (long low / short high).
    """

    def __init__(self, factor_weights=None, negate_composite=True, **kwargs):
        self.factor_weights = factor_weights or DEFAULT_FACTOR_WEIGHTS
        self.negate_composite = negate_composite

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Work on a copy to avoid mutating raw data
        df = df.copy()

        # Flatten multi-level columns if present: ('feature', 'HIGH') -> 'HIGH'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Helper: cross-sectional rank (pct) per datetime
        def cs_rank(series: pd.Series) -> pd.Series:
            return series.groupby("datetime", group_keys=False).rank(pct=True)

        # Helper: per-instrument rolling corr
        def ts_corr(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
            combined = pd.DataFrame({"a": s1, "b": s2}, index=s1.index)
            # groupby instrument (second level), rolling corr
            inst_level = combined.index.names[1] if len(combined.index.names) > 1 else 1
            return combined.groupby(level=inst_level, group_keys=False).apply(
                lambda g: g["a"].rolling(window, min_periods=window).corr(g["b"])
            )

        # --- Compute 8 factors ---
        factors = {}

        # alpha044: corr(high, cs_rank(volume), 5)
        factors["alpha044"] = ts_corr(df["HIGH"], cs_rank(df["VOLUME"]), 5)

        # alpha033: cs_rank(1 - open/close)
        factors["alpha033"] = cs_rank(df["A033_RAW"])

        # alpha017: cs_rank(ts_rank_close) * cs_rank(delta_delta_close) * cs_rank(ts_rank_vol_ratio)
        factors["alpha017"] = cs_rank(df["A017_P1"]) * cs_rank(df["A017_P2"]) * cs_rank(df["A017_P3"])

        # alpha026: fully time-series, already computed
        factors["alpha026"] = df["A026"]

        # alpha055: corr(cs_rank(RSV), cs_rank(volume), 6)
        factors["alpha055"] = ts_corr(cs_rank(df["A055_RSV"]), cs_rank(df["VOLUME"]), 6)

        # alpha002: corr(cs_rank(log_vol_delta), cs_rank(return), 6)
        factors["alpha002"] = ts_corr(cs_rank(df["A002_LOGVD"]), cs_rank(df["A002_RET"]), 6)

        # alpha039: cs_rank(d7 * (1 - cs_rank(wma))) * (1 + cs_rank(mr))
        factors["alpha039"] = cs_rank(df["A039_D7"] * (1 - cs_rank(df["A039_WMA"]))) * (1 + cs_rank(df["A039_MR"]))

        # alpha003: corr(cs_rank(open), cs_rank(volume), 10)
        factors["alpha003"] = ts_corr(cs_rank(df["OPEN"]), cs_rank(df["VOLUME"]), 10)

        # --- CS z-score each factor and weighted composite ---
        composite = pd.Series(0.0, index=df.index)
        for name, series in factors.items():
            weight = self.factor_weights.get(name, 0.0)
            if weight == 0.0:
                continue
            # Cross-sectional z-score per bar
            grouped = series.groupby("datetime", group_keys=False)
            mu = grouped.transform("mean")
            sigma = grouped.transform("std")
            sigma = sigma.replace(0, np.nan)
            z = (series - mu) / sigma
            z = z.clip(-3, 3).fillna(0)

            df[name] = z
            composite = composite + weight * z

        # Negate for qlib convention: high composite = buy candidate
        if self.negate_composite:
            composite = -composite

        df["COMPOSITE"] = composite

        # Keep only COMPOSITE + individual factor z-scores for downstream
        keep_cols = ["COMPOSITE"] + [n for n in factors if n in df.columns]
        result = df[keep_cols]

        # Restore multi-level columns: ('feature', col_name)
        result.columns = pd.MultiIndex.from_tuples(
            [("feature", c) for c in result.columns]
        )
        return result

    def is_for_infer(self) -> bool:
        return True

    def readonly(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# DataHandler
# ---------------------------------------------------------------------------
class Alpha101FMZHandler(DataHandlerLP):
    """
    FMZ Alpha101 8-factor DataHandler for crypto perpetual futures.

    YAML-configurable parameters:
        instruments, start_time, end_time, freq, factor_weights, negate_composite
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
        factor_weights=None,
        negate_composite=True,
        **kwargs,
    ):
        # Build infer processors: Alpha101FMZComposite first, then standard cleanup
        if infer_processors is None:
            infer_processors = [
                {
                    "class": "Alpha101FMZComposite",
                    "module_path": "qlib.contrib.data.handler_alpha101_fmz",
                    "kwargs": {
                        "factor_weights": factor_weights or DEFAULT_FACTOR_WEIGHTS,
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
            **kwargs,
        )

    @staticmethod
    def get_feature_config():
        return [_TS_FEATURE_EXPRS, _TS_FEATURE_NAMES]

    @staticmethod
    def get_label_config():
        # 4h crypto: next-bar return
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
