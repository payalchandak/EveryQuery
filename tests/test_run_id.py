"""Tests for the run_id resolver: ensures output_dir and wandb name share the same timestamp."""

import re
import time

import every_query.train as train_module


def test_run_id_format():
    """run_id() returns a string matching YYYY-MM-DD/HH-MM-SS."""
    train_module._RUN_ID = None  # reset cached value
    try:
        rid = train_module.run_id()
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}/\d{2}-\d{2}-\d{2}", rid), f"Unexpected format: {rid}"
    finally:
        train_module._RUN_ID = None


def test_run_id_is_stable():
    """Consecutive calls return the same value (singleton behaviour)."""
    train_module._RUN_ID = None
    try:
        first = train_module.run_id()
        time.sleep(1.1)  # ensure wall-clock moves past the second boundary
        second = train_module.run_id()
        assert first == second, f"run_id changed between calls: {first!r} vs {second!r}"
    finally:
        train_module._RUN_ID = None


def test_run_id_resets():
    """After clearing _RUN_ID, a new value is generated."""
    train_module._RUN_ID = None
    try:
        first = train_module.run_id()
        train_module._RUN_ID = None
        time.sleep(1.1)
        second = train_module.run_id()
        assert first != second, "run_id should differ after reset + time change"
    finally:
        train_module._RUN_ID = None
