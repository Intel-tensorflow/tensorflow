#!/usr/bin/env python3
"""
Validate the wheel file size and sure the wheel file size is smaller than
the limitation

Ex: ../validate_wheel_size.py <wheel_file_path/wheel_files_directory> [--limit size=170]
This will check the size of the wheel file should not great than the limit MB.

Unit test:
Execute the script without any arguments.
"""
import argparse
import doctest
import sys

from typing import Optional, Union
from pathlib import Path


class FileSizeError(Exception):
    def __init__(self, filepath: str, limit: int, *args, **kwargs):
        super().__init__(
            f"The wheel file {filepath} is over the limit {limit}MB",
            *args,
            **kwargs)


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args("c:/abc --limit 130")
    >>> p.path == Path("c:/abc")
    True
    >>> p.limit
    130
    >>> p = parse_args("unexist")
    >>> p.path == Path("unexist")
    True
    >>> p.limit
    170
    """
    p = argparse.ArgumentParser(description="Validate python wheel file size")
    p.add_argument("path",
                   type=Path,
                   help="Wheel file path or the directory contain wheel files")
    p.add_argument("--limit",
                   "-l",
                   type=int,
                   default=170,
                   required=False,
                   help="Limited wheel file MB size, default is 170MB")

    if test_args:
        return p.parse_args(test_args.split())
    else:
        return p.parse_args()


def _validate_file_size(path: Path, limit: int):
    """
    Validate the given file size is small or equal to the limit size.

    Args:
        path (Path): file absolute path
        limit (int): the limit MB for the file

    Raises:
        FileSizeError: If the file size is greater than the limit, an exception
        will be raised.

    >>> from tempfile import NamedTemporaryFile
    >>> import os
    >>> f = NamedTemporaryFile(delete=False)
    >>> # 0MB temp file
    >>> _validate_file_size(Path(f.name), 3)
    The wheel file ...
    >>> # 5MB temp file
    >>> f.write(b'1'*1024*1024*5)
    5242880
    >>> _validate_file_size(Path(f.name), 3)
    Traceback (most recent call last):
    FileSizeError: The wheel file ...
    >>> f.close()
    >>> os.remove(f.name)
    """
	limit = max(limit, 0)
    limit_bytes = limit * 1024 * 1024
    if path.stat().st_size > limit_bytes:
        raise FileSizeError(path, limit)
    else:
        print(f"The wheel file {path} is within the limit {limit}MB.")


def main(path: Union[Path, str], limit: int):
    """
    Main entry to validate the wheel file size

    Args:
        path (Path): wheel file absolute path or a directory contain the wheel files
        limit (int): the limit MB for the wheel file

    Raises:
        ValueError: Raise when no wheel file exist in the directory
        ValueError: Raise when file or directory not exist
        FileSizeError: Raise when wheel file is greater than the limit size

    >>> from tempfile import mkdtemp
    >>> from os.path import join
    >>> from shutil import rmtree
    >>> temp = mkdtemp()
    >>> fixtures = [("a.whl", 1), ("b.whl", 2), ("t.txt", 3), ("0.whl", 0)]
    >>> for fixture in fixtures:
    ...     name, size = fixture
    ...     with open(join(temp, name), "wb") as f:
    ...         f.write(b'1' * 1024 * 1024 * size)
    ...
    1048576
    2097152
    3145728
    0
    >>> main(Path(join(temp, "0.whl")), -1)
    The limit ...
    The wheel file ...
    >>> main(join(temp, "t.txt"), 5)
    Traceback (most recent call last):
    ValueError: The wheel path ...
    >>> main(temp, 2)
    The wheel file ...
    The wheel file ...
    The wheel file ...
    >>> main(join(temp, "b.whl"), 1)
    Traceback (most recent call last):
    FileSizeError: The wheel file ...
    >>> rmtree(temp)
    """
    if isinstance(path, str):
        path = Path(path)

    if limit < 0:
        print(f"The limit {limit} is smaller than 0, it will be treated as 0.")
        limit = 0

    if path.is_file() and path.suffix == ".whl":
        _validate_file_size(path, limit)
    elif path.is_dir():
        wheels: tuple[Path] = tuple(path.glob("*.whl"))
        if wheels:
            for wheel in wheels:
                _validate_file_size(wheel, limit)
        else:
            raise ValueError(f"There is no wheel file exist in the path {path}!")
    else:
        raise ValueError("The wheel path or wheel files directory not exist!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for validate_wheel_size.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        parser = parse_args()
        main(parser.path, parser.limit)