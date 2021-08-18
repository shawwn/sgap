import pytest
import subprocess
import tempfile
import os
from pathlib import Path

import sgap.system
import sgap.git


def test_system():
    assert sgap.system.run("echo", "hi") == "hi"
    assert sgap.system.run("cat", stdin="hi") == "hi"
    with pytest.raises(subprocess.CalledProcessError):
        sgap.system.run("bash", "-c", "exit 1", verbose=False)


def test_git():
    with tempfile.NamedTemporaryFile(dir=os.getcwd()) as f:
        assert Path(f.name).relative_to(os.getcwd()) in sgap.git.untracked_files()
    assert len(sgap.git.unstaged_files()) >= 0
