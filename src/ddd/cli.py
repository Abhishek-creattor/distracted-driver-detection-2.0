from __future__ import annotations

import argparse

from ddd.calibration import calibrate_driver
from ddd.config import load_config
from ddd.monitor import monitor_driver
from ddd.register import register_driver


def register_cli() -> None:
    parser = argparse.ArgumentParser(description="Register a new driver")
    parser.add_argument("--driver-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    settings = load_config(args.config)
    register_driver(settings, args.driver_id, args.name)


def calibrate_cli() -> None:
    parser = argparse.ArgumentParser(description="Calibrate EAR/MAR thresholds for a driver")
    parser.add_argument("--driver-id", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    settings = load_config(args.config)
    calibrate_driver(settings, args.driver_id)


def monitor_cli() -> None:
    parser = argparse.ArgumentParser(description="Run live distracted driver monitoring")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    settings = load_config(args.config)
    monitor_driver(settings)
