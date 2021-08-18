from pathlib import Path

from .. import system


def unstaged_files(**kwargs):
    return [
        Path(path)
        for path in system.run(
            *"git diff-files --name-only".split(), **kwargs
        ).splitlines()
    ]


def untracked_files(**kwargs):
    return [
        Path(path)
        for path in system.run(
            *"git ls-files --others --exclude-standard".split(), **kwargs
        ).splitlines()
    ]
