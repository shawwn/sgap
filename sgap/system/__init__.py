import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import shlex
import sys

from .. import util


def run(
    cmd: str,
    *args: str,
    stdin=None,
    verbose=True,
    strip=True,
    encoding="utf-8",
    error_output=sys.stderr,
    shell=False
):
    if shell:
        if args:
            cmd += " " + shlex.join(args)
    else:
        cmd = [cmd] + list(args)
    del args
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin is not None else None,
        encoding=encoding,
        shell=shell,
    )
    stdout, stderr = p.communicate(input=stdin)
    stdout = util.convert_string(stdout)
    stderr = util.convert_string(stderr)
    if p.returncode == 0:
        if stderr:
            if verbose:
                print("`%s` printed to stderr:" % cmd, file=error_output)
            print(stderr.rstrip(), file=error_output)
        if strip:
            stdout = stdout.rstrip("\r\n")
        return stdout
    if verbose:
        print("`%s` returned %s" % (cmd, p.returncode), file=error_output)
    if stderr:
        print(stderr.rstrip(), file=error_output)
    raise subprocess.CalledProcessError(p.returncode, cmd, output=stdout, stderr=stderr)
