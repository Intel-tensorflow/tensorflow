#!/usr/bin/env python3
"""
A directory clean helper script

usage: clean.py [-h] --path PATH [PATH ...] [--dry]

Clean the given directories without raise errors. The root directories will be kept.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH [PATH ...], -p PATH [PATH ...]
                        The path to be removed. Can be passed multiple times
  --dry, -d             Dry run the clean operations
"""
import shutil
import argparse
import doctest
import sys
import os

from pathlib import Path
from typing import Optional, Iterable


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args(['-p', 'c:', '-p', 'd:', '-p', 'e:', '-d'])
    >>> p.path
    [WindowsPath('c:'), WindowsPath('d:'), WindowsPath('e:')]
    >>> p.dry
    True
    """
    p = argparse.ArgumentParser(
        description="Clean the given directories without raise errors. The root directories will be kept.")
    p.add_argument("--path",
                   "-p",
                   action="extend",
                   required=True,
                   nargs="+",
                   type=Path,
                   help="The path to be removed. Can be passed multiple times")
    p.add_argument("--dry",
                   "-d",
                   action="store_true",
                   default=False,
                   required=False,
                   help="Dry run the clean operations")

    if isinstance(test_args, str):
        return p.parse_args(test_args.split())
    elif isinstance(test_args, list):
        return p.parse_args(test_args)
    else:
        return p.parse_args()


def safe_remove(is_dry: bool, path: Path):
    try:
        if is_dry:
            print(f"Dry run remove {'directory' if path.is_dir() else 'file'} {path}")
        else:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    except Exception as e:
        print(f"Safe remove {path} failed with error {e}", file=sys.stderr)


def remove_subitems(is_dry: bool, path: Path):
    if path.is_dir():
        for item in os.listdir(path):
            safe_remove(is_dry, path.joinpath(item))
    else:
        safe_remove(is_dry, path)


def main(is_dry: bool, *args: Iterable[Path]):
    """
    The main entry of the clean method

    Args:
        is_dry (bool): Is dry run the clean operation
    >>> main(True, *[Path('c'), Path('d')])
    Dry run remove paths ...
    """
    if is_dry:
        print(f"Dry run remove paths {args}")

    for path in args:
        remove_subitems(is_dry, path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for clean.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        args = parse_args()
        main(args.dry,
             *args.path)
