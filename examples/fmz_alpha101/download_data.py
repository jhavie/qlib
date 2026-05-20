"""
Config-driven data download for FMZ Alpha101 strategy.

Usage:
    # Download + normalize + dump (full pipeline)
    python examples/fmz_alpha101/download_data.py

    # Custom config path
    python examples/fmz_alpha101/download_data.py --config examples/fmz_alpha101/data_config.yaml

    # Only download (skip normalize/dump)
    python examples/fmz_alpha101/download_data.py --step download

    # Only normalize (requires source CSVs already present)
    python examples/fmz_alpha101/download_data.py --step normalize

    # Only dump to bin (requires normalized CSVs already present)
    python examples/fmz_alpha101/download_data.py --step dump
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruamel.yaml import YAML
from loguru import logger

from scripts.data_collector.binance_um.collector import Run


def load_config(config_path: str) -> dict:
    yaml = YAML()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f)["data_collection"]


def run_download(cfg: dict) -> None:
    """Download raw kline data from Binance REST API."""
    symbols = ",".join(cfg["symbols"])
    runner = Run(
        source_dir=cfg["source_dir"],
        normalize_dir=cfg["normalize_dir"],
        interval=cfg["interval"],
    )
    logger.info(f"Downloading {len(cfg['symbols'])} symbols, interval={cfg['interval']}, {cfg['start']} ~ {cfg['end']}")
    runner.download_data(
        start=cfg["start"],
        end=cfg["end"],
        symbols=symbols,
        delay=cfg.get("delay", 0.3),
    )
    logger.info("Download completed.")


def run_normalize(cfg: dict) -> None:
    """Normalize downloaded CSVs into qlib-ready schema."""
    runner = Run(
        source_dir=cfg["source_dir"],
        normalize_dir=cfg["normalize_dir"],
        interval=cfg["interval"],
    )
    logger.info("Normalizing data...")
    runner.normalize_data(
        date_field_name="date",
        symbol_field_name="symbol",
        fill_missing=True,
    )
    logger.info("Normalize completed.")


def run_dump(cfg: dict) -> None:
    """Dump normalized CSVs to qlib binary format."""
    runner = Run(
        source_dir=cfg["source_dir"],
        normalize_dir=cfg["normalize_dir"],
        interval=cfg["interval"],
    )
    logger.info(f"Dumping to qlib binary: {cfg['qlib_dir']}")
    runner.dump_to_bin(qlib_dir=cfg["qlib_dir"])
    logger.info("Dump completed.")


def main():
    parser = argparse.ArgumentParser(description="FMZ Alpha101 data download pipeline")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "examples" / "fmz_alpha101" / "data_config.yaml"),
        help="Path to data_config.yaml",
    )
    parser.add_argument(
        "--step",
        choices=["all", "download", "normalize", "dump"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ("all", "download"):
        run_download(cfg)
    if args.step in ("all", "normalize"):
        run_normalize(cfg)
    if args.step in ("all", "dump"):
        run_dump(cfg)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
