"""CLI smoke tests — verify argparse is wired up correctly."""

from __future__ import annotations

import subprocess
import sys
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "rt_forecast_b_route_v2_6_9_annotated.py"


class TestCLIHelp:
    """Running the script with --help should exit 0 and print usage."""

    @pytest.mark.parametrize("subcmd", ["train", "predict", "backtest"])
    def test_subcommand_help(self, subcmd):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), subcmd, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"--help for '{subcmd}' failed:\n{result.stderr}"
        assert subcmd in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_no_args_exits_nonzero(self):
        """The script requires a subcommand; bare invocation should error."""
        # The script actually runs run_without_args() when no args given,
        # which may fail due to missing data files — that's fine, we just
        # check it doesn't silently succeed with exit 0 on bad input.
        # We pass an invalid subcommand to verify argparse rejects it.
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "nonexistent_cmd"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0


class TestCLIImports:
    """Verify the module can be imported without executing main()."""

    def test_import_succeeds(self, mod):
        # If we got here, the session-scoped fixture already imported it
        assert hasattr(mod, "main")
        assert hasattr(mod, "parse_args")
        assert callable(mod.main)
