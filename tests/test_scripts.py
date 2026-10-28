import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("bash") is None, reason="coverage script requires bash")
@pytest.mark.parametrize("status", [0, 1, 5])
def test_coverage_script_preserves_test_exit_status(tmp_path, status):
    script = Path(__file__).resolve().parents[1] / "scripts" / "coverage.sh"
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text("")
    pytest_command = bin_dir / "pytest"
    pytest_command.write_text(f"#!/bin/sh\necho 'test output'\nexit {status}\n")
    pytest_command.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", str(script), "arrayfile"], cwd=tmp_path, env=env, capture_output=True, check=False)
    assert result.returncode == status
    assert (tmp_path / "htmlcov" / "coverage_report.txt").read_text() == "test output\n"
