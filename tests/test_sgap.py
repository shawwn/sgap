import sgap.system
import pytest
import subprocess


def test_system():
    assert sgap.system.run("echo", "hi") == "hi"
    assert sgap.system.run("cat", stdin="hi") == "hi"
    with pytest.raises(subprocess.CalledProcessError):
        sgap.system.run("bash", "-c", "exit 1", verbose=False)
