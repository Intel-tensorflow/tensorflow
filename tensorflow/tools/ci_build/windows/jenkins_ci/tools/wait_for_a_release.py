import argparse
import doctest
import sys

from urllib import request
from xml.etree import ElementTree
from time import sleep, time
from typing import Optional
from collections.abc import Iterable


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args(['-p','tf', '-v', '2.0rc', '-t', '3'])
    >>> p.package
    'tf'
    >>> p.version
    '2.0rc'
    >>> p.timeout
    3
    >>> p = parse_args(['-p','tf', '-v', '2.0rc'])
    >>> p.timeout
    0
    """
    p = argparse.ArgumentParser(
        description="Wait for a specific python package version from https://pypi.org/rss/project/package/releases.xml")
    p.add_argument("--package",
                   "-p",
                   type=str,
                   required=True,
                   help="The python package name")
    p.add_argument("--version",
                   "-v",
                   type=str,
                   required=True,
                   help="The expected version of the package")
    p.add_argument("--timeout",
                   "-t",
                   type=int,
                   default=0,
                   required=False,
                   help="The timeout hours, if not provided their will be no timeout when the wait flag is provided.")

    if isinstance(test_args, str):
        return p.parse_args(test_args.split())
    elif isinstance(test_args, list):
        return p.parse_args(test_args)
    else:
        return p.parse_args()


def get_released_versions(package: str) -> Iterable:
    r = request.Request(
        f'https://pypi.org/rss/project/{package}/releases.xml', method='GET')
    rp = request.urlopen(r)
    rss = ElementTree.fromstring(rp.read())
    return tuple(x.text for x in rss.findall(".//item/title"))


def main(package: str, version: str, timeout: int):
    start = time()
    print(f'Waiting for {package} version {version} on pypi.python.org with timeout {timeout} hours')
    while True:
        versions = get_released_versions(package)
        if version in versions:
            print('The expected package version is found on pypi.org!')
            return
        elif (time() - start) / 3600 > timeout:
            print('Warning: Wait for the package release timeout!')
            return
        else:
            sleep(60)
            continue


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for wait_for_release.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        args = parse_args()
        main(args.package,
             args.version,
             args.timeout)
