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
from glob import glob


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args(['-p', 'c:', '-p', 'd:', '-p', 'e:', '-d', '-g', 'c:\\t*', '-r', 'c:/a*', '-r', 'd:/b*'])
    >>> p.path
    [WindowsPath('c:'), WindowsPath('d:'), WindowsPath('e:')]
    >>> p.dry
    True
    >>> p.glob
    ['c:\\t*']
    >>> p.rd
    ['c:/a*', 'd:/b*']
    """
    p = argparse.ArgumentParser(
        description="Clean the given directories without raise errors. The root directories will be kept.")
    p.add_argument("--path",
                   "-p",
                   action="extend",
                   required=False,
                   default=[],
                   nargs="*",
                   type=Path,
                   help="The path of the sub items to be removed. Can be passed multiple times")
    p.add_argument("--glob",
                   "-g",
                   action="extend",
                   required=False,
                   type=str,
                   default=[],
                   nargs="*",
                   help="The unix glob path pattern to search the clean directories. Support *, ?, [0-9], **")
    p.add_argument("--rd",
                   "-r",
                   action="extend",
                   type=str,
                   default=[],
                   nargs="*",
                   help="Remove the directories match the glob pattern instead of remove the sub items.")
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
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

        if is_dry:
            print("Dry run remove {path}")
        else:
            print("Removed {path}")

    except Exception as e:
        print(f"Safe remove {path} failed with error {e}", file=sys.stderr)


def remove_subitems(is_dry: bool, path: Path):
    if is_dry:
        print(f"Dry run remove {'directory' if path.is_dir() else 'file'} {path} and it is exist {path.exists()}")
    else:
        print(f"Removing {'directory subitems' if path.is_dir() else 'file'} {path}")
        if path.is_dir():
            for item in os.listdir(path):
                safe_remove(is_dry, path.joinpath(item))
        else:
            safe_remove(is_dry, path)


def remove_glob_subitems(is_dry: bool, glob_path: [str, Path], is_remove_root: bool):
    try:
        paths = (Path(p) for p in glob(str(glob_path)))
    except Exception as e:
        print(f"Glob file path failed with error, {e}", file=sys.stderr)
        return

    if is_remove_root:
        for path in paths:
            print(f"Removing {'directory ' if path.is_dir() else 'file'} {path}")
            safe_remove(is_dry, path)
    else:
        for path in paths:
            remove_subitems(is_dry, path)


def main(is_dry: bool, paths: Iterable[Path], glob_paths: Iterable[str], rd_paths: Iterable[str]):
    """
    The main entry of the clean method

    Args:
        is_dry (bool): Is dry run the clean operation
        paths (Iterable[Path]): The paths to clean
        glob_paths (Iterable[str]): The glob path pattern
    >>> main(True, [Path('c'), Path('d')], [], [])
    Dry run remove paths ...
    >>> import tempfile
    >>> tmp = tempfile.TemporaryDirectory()
    >>> tmp_path = Path(tmp.name)
    >>> a1 = Path.joinpath(tmp_path, 'a1')
    >>> a2 = Path.joinpath(tmp_path, 'a2')
    >>> os.mkdir(a1)
    >>> os.mkdir(a2)
    >>> t1 = Path(Path.joinpath(tmp_path, 'a1', 't1.txt'))
    >>> t2 = Path(Path.joinpath(tmp_path, 'a2', 't2.txt'))
    >>> open(t1, 'w').close()
    >>> open(t2, 'w').close()
    >>> t1.exists()
    True
    >>> t2.exists()
    True
    >>> main(False, [], [Path.joinpath(tmp_path, 'a*')], [Path.joinpath(tmp_path, 'a2')])
    Remove directory...
    Remove directory...
    Remove directory...
    >>> t1.exists()
    False
    >>> t2.exists()
    False
    >>> a1.exists()
    True
    >>> a2.exists()
    False
    >>> main(False, [tmp_path], [], [])
    Remove directory...
    >>> a1.exists()
    False
    >>> tmp.cleanup()
    """
    if is_dry:
        print(f"Dry run remove paths {paths}, {glob_paths}")

    for path in paths:
        remove_subitems(is_dry, path)

    for glob_path in glob_paths:
        remove_glob_subitems(is_dry, glob_path, False)

    for rd_path in rd_paths:
        remove_glob_subitems(is_dry, rd_path, True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for clean.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        args = parse_args()
        main(args.dry,
             args.path,
             args.glob,
             args.rd)
