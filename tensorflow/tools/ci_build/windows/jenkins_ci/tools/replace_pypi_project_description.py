#!/usr/bin/env python3
import argparse
import doctest
import sys
import base64


from pathlib import Path
from typing import Optional


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args("c:/abc -d abcd")
    >>> p.path == Path("c:/abc")
    True
    >>> p.description
    'abcd'
    >>> p = parse_args("unexist -d x")
    >>> p.path == Path("unexist")
    True
    """
    p = argparse.ArgumentParser(description="Validate python wheel file size")
    p.add_argument("path",
                   type=Path,
                   help="The path to the tensorflow package file setup.py")
    p.add_argument("--description",
                   "-d",
                   required=True,
                   help="Base64 encoded description")

    if test_args:
        return p.parse_args(test_args.split())
    else:
        return p.parse_args()


def replace_pypi_description(path, description):
    """Replace the script document string with the description

    Args:
        path (Path): The path to the package script setup.py
        description (str): The new pypi description

    >>> s = ["# abc",
    ...      '\"\"\" def',
    ...      '\"\"\"',
    ...      "end"]
    >>> from tempfile import NamedTemporaryFile
    >>> from os import remove
    >>> f = NamedTemporaryFile(delete=False, mode='w')
    >>> f.write('\\n'.join(s))
    2...
    >>> path = f.name
    >>> f.close()
    >>> open(path).read()
    '# abc\\n\"\"\" def\\n\"\"\"\\nend'
    >>> replace_pypi_description(path, " ghi")
    >>> open(path).read()
    '# abc\\n\"\"\" ghi\"\"\"\\nend'
    >>> remove(path)
    """
    with open(path) as f:
        lines = f.readlines()
        start, end = 0, 0
        for i in range(len(lines)):
            if lines[i].startswith('"""'):
                if start == 0:
                    start = i
                elif end == 0:
                    end = i
                elif start != 0 and end == 0:
                    break

    with open(path, "w") as f:
        is_replaced = False
        for i in range(len(lines)):
            if(i < start or i > end):
                f.write(lines[i])
            elif not is_replaced:
                is_replaced = True
                f.write(f'"""{description}"""\n')


def main(path, base64Description):
    description = base64.b64decode(base64Description).decode("UTF-8")
    replace_pypi_description(path, description)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"*** Run doctest for {__file__}! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        parser = parse_args()
        main(parser.path, parser.description)
