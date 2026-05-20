"""
Config-driven Model-Free FMZ Alpha101 backtest runner.

This script:
1. Reads workflow_model_free.yaml for all configuration
2. Creates Alpha101FMZHandler to compute the 8-factor COMPOSITE signal
3. Uses the COMPOSITE directly (no ML training) as the prediction signal
4. Runs LongShortTopKStrategy + ShortableExecutor backtest

Usage:
    python examples/fmz_alpha101/run_model_free.py
    python examples/fmz_alpha101/run_model_free.py --config examples/fmz_alpha101/workflow_model_free.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import qlib
from ruamel.yaml import YAML
from loguru import logger
from qlib.constant import REG_CRYPTO
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord


def load_config(config_path: str) -> dict:
    yaml = YAML()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f)


def _build_strategy_config(strat_cfg: dict, pred_signal) -> dict:
    """Build strategy config from YAML, supporting both LongShortTopKStrategy and FMZNeutralStrategy."""
    strat_class = strat_cfg.get("class", "LongShortTopKStrategy")
    strat_module = strat_cfg.get("module_path", "qlib.contrib.strategy.signal_strategy")

    # Build kwargs based on strategy class
    strat_kwargs = {"signal": pred_signal}

    if strat_class == "FMZNeutralStrategy":
        strat_kwargs.update({
            "topk_long": strat_cfg["topk_long"],
            "topk_short": strat_cfg["topk_short"],
            "value_per_position": strat_cfg["value_per_position"],
        })
    else:
        # LongShortTopKStrategy (default)
        strat_kwargs.update({
            "topk_long": strat_cfg["topk_long"],
            "topk_short": strat_cfg["topk_short"],
            "n_drop_long": strat_cfg["n_drop_long"],
            "n_drop_short": strat_cfg["n_drop_short"],
            "hold_thresh": strat_cfg.get("hold_thresh", 1),
            "only_tradable": strat_cfg.get("only_tradable", True),
            "forbid_all_trade_at_limit": False,
            "rebalance_to_weights": strat_cfg.get("rebalance_to_weights", False),
        })

    return {
        "class": strat_class,
        "module_path": strat_module,
        "kwargs": strat_kwargs,
    }


def _sym_to_nautilus(instrument: str) -> str:
    """Convert qlib instrument id (e.g. 'BINANCE_UM.BTCUSDT') to nautilus format ('BTCUSDT.BINANCE')."""
    sym = instrument.upper()
    if sym.startswith("BINANCE_UM."):
        sym = sym[len("BINANCE_UM."):]
    elif "." in sym:
        sym = sym.split(".")[-1]
    return f"{sym}.BINANCE"


def _ts_to_ns_epoch(ts) -> int:
    """Convert a datetime/Timestamp to nanosecond epoch int."""
    return int(pd.Timestamp(ts).timestamp() * 1e9)


def _ts_to_utc_str(ts) -> str:
    """Convert a datetime/Timestamp to UTC string with +00:00 suffix."""
    s = str(pd.Timestamp(ts))
    if "+" not in s and "Z" not in s:
        s += "+00:00"
    return s


def export_positions_csv(
    positions_dict: dict,
    output_path: Path,
    open_cost: float = 0.0002,
    close_cost: float = 0.0002,
    deal_price: str = "open",
    freq: str = "240min",
) -> None:
    """Generate a nautilus-aligned positions_report.csv from qlib position snapshots.

    Detects position opens/closes across consecutive bars and writes one row
    per closed position with headers matching nautilus positions_report.csv.

    Uses raw OHLCV deal_price (e.g. bar open) instead of mark-to-market close
    for accurate avg_px_open / avg_px_close pricing.
    """
    # --- Load raw deal prices for accurate entry/exit pricing ---
    from qlib.data import D

    all_instruments: set = set()
    for pos_obj in positions_dict.values():
        for inst in pos_obj.position:
            if inst not in ("cash", "now_account_value", "cash_delay"):
                all_instruments.add(inst)

    sorted_items = sorted(positions_dict.items())
    timestamps = [ts for ts, _ in sorted_items]

    price_field = f"${deal_price}"  # e.g. "$open"
    raw_prices = D.features(
        list(all_instruments),
        fields=[price_field],
        start_time=str(timestamps[0]),
        end_time=str(timestamps[-1]),
        freq=freq,
    )
    # raw_prices: MultiIndex (instrument, datetime) → price column
    price_col = raw_prices.columns[0]  # the actual column name after qlib rename

    def _get_deal_price(inst: str, ts) -> float | None:
        """Look up the deal price for instrument at timestamp."""
        try:
            val = raw_prices.loc[(inst, ts), price_col]
            if pd.notna(val):
                return float(val)
        except KeyError:
            pass
        return None

    # Track live holdings: instrument → {amount, price, open_ts, open_px}
    prev_holdings: dict = {}
    closed_positions: list[dict] = []
    pos_counter: dict[str, int] = {}

    for ts, pos_obj in sorted_items:
        # Extract per-instrument holdings for this bar
        curr_holdings: dict = {}
        for inst, info in pos_obj.position.items():
            if inst in ("cash", "now_account_value", "cash_delay"):
                continue
            curr_holdings[inst] = {
                "amount": info["amount"],
                "price": info["price"],
            }

        # --- Detect closes: was in prev but gone or sign flipped ---
        for inst, prev in prev_holdings.items():
            curr = curr_holdings.get(inst)
            sign_flipped = curr is not None and prev["amount"] * curr["amount"] < 0
            disappeared = curr is None or abs(curr.get("amount", 0)) < 1e-9

            if disappeared or sign_flipped:
                amt = abs(prev["amount"])
                # Use raw deal_price for open/close instead of mark-to-market
                px_open = _get_deal_price(inst, prev["open_ts"])
                if px_open is None:
                    px_open = prev["open_px"]  # fallback
                px_close = _get_deal_price(inst, ts)
                if px_close is None:
                    px_close = prev["price"]  # fallback

                direction = 1 if prev["amount"] > 0 else -1
                comm = amt * px_open * open_cost + amt * px_close * close_cost
                raw_pnl = (px_close - px_open) * amt * direction
                realized_pnl = raw_pnl - comm
                notional_open = amt * px_open
                realized_return = realized_pnl / notional_open if notional_open > 0 else 0.0

                open_ts = prev["open_ts"]
                close_ts = ts
                duration_ns = float((close_ts - open_ts).total_seconds() * 1e9)

                n = pos_counter.get(inst, 0) + 1
                pos_counter[inst] = n

                closed_positions.append({
                    "position_id": f"QLIB-{inst}-{n:03d}",
                    "trader_id": "QLIB-BACKTESTER",
                    "strategy_id": "Alpha044FMZ",
                    "instrument_id": _sym_to_nautilus(inst),
                    "account_id": "QLIB-001",
                    "opening_order_id": "",
                    "closing_order_id": "",
                    "entry": "BUY" if prev["amount"] > 0 else "SELL",
                    "side": "FLAT",
                    "quantity": "0.00000000",
                    "peak_qty": f"{amt:.8f}",
                    "ts_init": _ts_to_ns_epoch(open_ts),
                    "ts_opened": _ts_to_utc_str(open_ts),
                    "ts_last": _ts_to_ns_epoch(close_ts),
                    "ts_closed": _ts_to_utc_str(close_ts),
                    "duration_ns": duration_ns,
                    "avg_px_open": f"{px_open:.8f}",
                    "avg_px_close": f"{px_close:.8f}",
                    "commissions": f"['{comm:.8f} USDT']",
                    "realized_return": f"{realized_return:.8f}",
                    "realized_pnl": f"{realized_pnl:.8f} USDT",
                    "is_snapshot": False,
                })

        # --- Detect opens: new instrument or sign flip ---
        for inst, curr in curr_holdings.items():
            prev = prev_holdings.get(inst)
            sign_flipped = prev is not None and prev["amount"] * curr["amount"] < 0
            is_new = prev is None or abs(prev.get("amount", 0)) < 1e-9

            if is_new or sign_flipped:
                curr["open_ts"] = ts
                # Use raw deal_price for open instead of mark-to-market
                dp = _get_deal_price(inst, ts)
                curr["open_px"] = dp if dp is not None else curr["price"]
            else:
                # Carry forward open info from previous bar
                curr["open_ts"] = prev["open_ts"]
                curr["open_px"] = prev["open_px"]

        # --- Update tracking (only keep instruments with material positions) ---
        prev_holdings = {
            k: v for k, v in curr_holdings.items() if abs(v["amount"]) >= 1e-9
        }

    # --- Handle still-open positions at end: write as snapshots ---
    for inst, hold in prev_holdings.items():
        amt = abs(hold["amount"])
        px_open = hold["open_px"]
        px_last = hold["price"]
        comm = amt * px_open * open_cost  # only open-side commission so far

        n = pos_counter.get(inst, 0) + 1
        pos_counter[inst] = n

        last_ts = sorted_items[-1][0]

        closed_positions.append({
            "position_id": f"QLIB-{inst}-{n:03d}",
            "trader_id": "QLIB-BACKTESTER",
            "strategy_id": "Alpha044FMZ",
            "instrument_id": _sym_to_nautilus(inst),
            "account_id": "QLIB-001",
            "opening_order_id": "",
            "closing_order_id": "",
            "entry": "BUY" if hold["amount"] > 0 else "SELL",
            "side": "LONG" if hold["amount"] > 0 else "SHORT",
            "quantity": f"{amt:.8f}",
            "peak_qty": f"{amt:.8f}",
            "ts_init": _ts_to_ns_epoch(hold["open_ts"]),
            "ts_opened": _ts_to_utc_str(hold["open_ts"]),
            "ts_last": _ts_to_ns_epoch(last_ts),
            "ts_closed": "",
            "duration_ns": "",
            "avg_px_open": f"{px_open:.8f}",
            "avg_px_close": f"{px_last:.8f}",
            "commissions": f"['{comm:.8f} USDT']",
            "realized_return": "0.00000000",
            "realized_pnl": "0.00000000 USDT",
            "is_snapshot": True,
        })

    df = pd.DataFrame(closed_positions)
    col_order = [
        "position_id", "trader_id", "strategy_id", "instrument_id",
        "account_id", "opening_order_id", "closing_order_id", "entry",
        "side", "quantity", "peak_qty", "ts_init", "ts_opened", "ts_last",
        "ts_closed", "duration_ns", "avg_px_open", "avg_px_close",
        "commissions", "realized_return", "realized_pnl", "is_snapshot",
    ]
    df = df.reindex(columns=col_order)
    df.to_csv(output_path, index=False)
    logger.info(f"Positions CSV ({len(df)} rows): {output_path}")


def export_account_csv(report_df: pd.DataFrame, output_path: Path) -> None:
    """Generate a per-bar account CSV from qlib report_normal DataFrame."""
    out = report_df[["account", "cash", "value", "total_cost", "return", "bench"]].copy()
    out.columns = ["total", "cash", "position_value", "total_cost", "return", "benchmark"]
    out.index.name = "datetime"
    out.to_csv(output_path)
    logger.info(f"Account CSV ({len(out)} rows): {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FMZ Alpha101 Model-Free backtest")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "examples" / "fmz_alpha101" / "workflow_model_free.yaml"),
        help="Path to workflow_model_free.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    # --- Init qlib ---
    qlib_cfg = cfg["qlib_init"]
    qlib.init(provider_uri=qlib_cfg["provider_uri"], region=REG_CRYPTO, kernels=1)

    # --- Build handler ---
    handler_kwargs = dict(cfg["data_handler_config"])
    handler_class = handler_kwargs.pop("class", "Alpha101FMZHandler")
    handler_module = handler_kwargs.pop("module_path", "qlib.contrib.data.handler_alpha101_fmz")
    handler_config = {
        "class": handler_class,
        "module_path": handler_module,
        "kwargs": handler_kwargs,
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": [handler_kwargs["fit_start_time"], handler_kwargs["fit_end_time"]],
                "valid": ["2024-01-01", "2024-06-30"],
                "test": [cfg["backtest"]["start_time"], cfg["backtest"]["end_time"]],
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)

    # --- Extract COMPOSITE signal from test segment ---
    # The handler's infer processor produces a DataFrame with 'COMPOSITE' column
    test_df = dataset.prepare("test", col_set="__all", data_key="infer")
    logger.info(f"Test data shape: {test_df.shape}")

    # Extract COMPOSITE as the prediction signal
    if isinstance(test_df.columns, __import__("pandas").MultiIndex):
        composite_col = [c for c in test_df.columns if c[-1] == "COMPOSITE"]
        if composite_col:
            pred_signal = test_df[composite_col[0]]
        else:
            raise ValueError(f"COMPOSITE column not found. Available: {test_df.columns.tolist()}")
    else:
        pred_signal = test_df["COMPOSITE"]

    pred_signal = pred_signal.to_frame("score")
    logger.info(f"Signal shape: {pred_signal.shape}, sample:\n{pred_signal.head(10)}")

    # --- Export factor values CSV for cross-system comparison ---
    try:
        factor_export = test_df.reset_index()
        if isinstance(factor_export.columns, pd.MultiIndex):
            factor_export.columns = [c[-1] for c in factor_export.columns]
        # Keep only instrument, datetime, COMPOSITE
        cols_to_keep = ["instrument", "datetime"]
        if "COMPOSITE" in factor_export.columns:
            cols_to_keep.append("COMPOSITE")
        else:
            logger.warning(f"COMPOSITE not in columns for factor export: {factor_export.columns.tolist()}")
        factor_export = factor_export[cols_to_keep].copy()
        factor_export["instrument"] = factor_export["instrument"].apply(_sym_to_nautilus)
        factor_export.rename(columns={"COMPOSITE": "composite"}, inplace=True)
        config_path_for_export = Path(args.config)
        factor_csv_path = config_path_for_export.parent / "qlib_factors.csv"
        factor_export.to_csv(factor_csv_path, index=False)
        logger.info(f"Factor values CSV ({len(factor_export)} rows): {factor_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to export factor values CSV: {e}")

    # --- Backtest with LongShortTopKStrategy ---
    bt_cfg = cfg["backtest"]
    strat_cfg = cfg["strategy"]
    exch_cfg = cfg["exchange"]

    # Prefer crypto PortAnaRecord
    try:
        from qlib.contrib.workflow.crypto_record_temp import CryptoPortAnaRecord as PortAnaRecord
        logger.info("Using CryptoPortAnaRecord")
    except Exception:
        from qlib.workflow.record_temp import PortAnaRecord
        logger.info("Using default PortAnaRecord")

    port_analysis_config = {
        "executor": {
            "class": "ShortableExecutor",
            "module_path": "qlib.backtest.shortable_backtest",
            "kwargs": {
                "time_per_step": exch_cfg["freq"],
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": _build_strategy_config(strat_cfg, pred_signal),
        "backtest": {
            "start_time": bt_cfg["start_time"],
            "end_time": bt_cfg["end_time"],
            "account": bt_cfg["account"],
            "benchmark": bt_cfg["benchmark"],
            "exchange_kwargs": {
                "exchange": {
                    "class": "ShortableExchange",
                    "module_path": "qlib.backtest.shortable_exchange",
                },
                "freq": exch_cfg["freq"],
                "limit_threshold": exch_cfg["limit_threshold"],
                "deal_price": exch_cfg["deal_price"],
                "open_cost": exch_cfg["open_cost"],
                "close_cost": exch_cfg["close_cost"],
                "min_cost": exch_cfg["min_cost"],
            },
        },
    }

    # --- Run experiment ---
    with R.start(experiment_name="fmz_alpha101_model_free"):
        recorder = R.get_recorder()

        # Save signal directly (model-free: no model.predict needed)
        recorder.save_objects(**{"pred.pkl": pred_signal})

        # Save label for SigAnaRecord dependency
        raw_label = SignalRecord.generate_label(dataset)
        if raw_label is not None:
            recorder.save_objects(**{"label.pkl": raw_label})

        # Signal analysis
        sar = SigAnaRecord(recorder, ana_long_short=True, ann_scaler=2190)
        sar.generate()

        # Portfolio analysis
        par = PortAnaRecord(recorder, port_analysis_config, exch_cfg["freq"])
        par.generate()

        # --- Load artifacts for reporting ---
        artifact_uri = recorder.get_artifact_uri()
        artifact_dir = Path(artifact_uri.replace("file://", ""))
        report_files = list(artifact_dir.glob("**/report_normal_*.pkl"))
        report_df = None
        if report_files:
            with open(report_files[0], "rb") as f:
                report_df = pickle.load(f)

        # --- Generate HTML report (account-based, aligned with FMZ) ---
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            if report_df is None:
                raise FileNotFoundError(f"No report_normal_*.pkl in {artifact_dir}")

            acct = report_df["account"]
            acct_ret = (acct / acct.iloc[0] - 1) * 100  # account growth %
            bench_ret = ((1 + report_df["bench"]).cumprod() - 1) * 100  # benchmark %
            drawdown = (acct / acct.cummax() - 1) * 100  # drawdown %
            total_cost = report_df["total_cost"]  # cumulative commission

            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=["Account Return (%)", "Drawdown (%)", "Turnover", "Cumulative Commission"],
            )
            fig.add_trace(go.Scatter(x=acct_ret.index, y=acct_ret, name="Strategy", line=dict(width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=bench_ret.index, y=bench_ret, name="Benchmark", line=dict(width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill="tozeroy", line=dict(width=1, color="red")), row=2, col=1)
            fig.add_trace(go.Bar(x=report_df.index, y=report_df["turnover"], name="Turnover", marker_color="rgba(100,100,200,0.4)"), row=3, col=1)
            fig.add_trace(go.Scatter(x=total_cost.index, y=total_cost, name="Cum. Commission", fill="tozeroy", line=dict(width=1.5, color="orange")), row=4, col=1)

            total_ret = acct_ret.iloc[-1]
            final_cost = total_cost.iloc[-1]
            cost_pct = final_cost / acct.iloc[0] * 100
            fig.update_layout(
                height=1000,
                title_text=f"Backtest Report — Return: {total_ret:.2f}% | Commission: {final_cost:.2f} ({cost_pct:.2f}%)",
                showlegend=True, template="plotly_white",
            )

            config_path = Path(args.config)
            html_path = config_path.parent / (config_path.stem + "_report.html")
            fig.write_html(str(html_path))
            logger.info(f"HTML report saved to: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")

        # --- Export CSV reports (aligned with nautilus format) ---
        try:
            positions_files = list(artifact_dir.glob("**/positions_normal_*.pkl"))
            if not positions_files:
                raise FileNotFoundError(f"No positions_normal_*.pkl in {artifact_dir}")
            with open(positions_files[0], "rb") as f:
                positions_dict = pickle.load(f)

            config_path = Path(args.config)
            csv_dir = config_path.parent
            positions_csv = csv_dir / (config_path.stem + "_positions.csv")
            account_csv = csv_dir / (config_path.stem + "_account.csv")

            export_positions_csv(
                positions_dict,
                positions_csv,
                open_cost=exch_cfg["open_cost"],
                close_cost=exch_cfg["close_cost"],
                deal_price=exch_cfg.get("deal_price", "open"),
                freq=exch_cfg.get("freq", "240min"),
            )
            if report_df is not None:
                export_account_csv(report_df, account_csv)
        except Exception as e:
            logger.warning(f"Failed to export CSV: {e}")

    logger.info("Model-Free backtest completed. Check mlruns/ for results.")


if __name__ == "__main__":
    main()
