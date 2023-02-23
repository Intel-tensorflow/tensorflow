#!/usr/bin/env python3
"""
Download TensorFlow wheel files from public Jenkins CI artifacts. It will
automatic parse the success build and download the wheel file from the artifact base on the job build history.

Help information:
usage: download_wheels.py [-h] --server SERVER --job JOB --version VERSION
                          [--py_version [PY_VERSION ...]] [--artifact [ARTIFACT ...]]
                          [--output OUTPUT] [--timeout TIMEOUT]

Wait for a specific python package version from
https://pypi.org/rss/project/package/releases.xml

optional arguments:
  -h, --help            show this help message and exit
  --server SERVER, -s SERVER
                        The Jenkins server url
  --job JOB, -j JOB     The job url to search
  --version VERSION, -v VERSION
                        The desire package version to search from the job history
  --py_version [PY_VERSION ...], -p [PY_VERSION ...]
                        The required package Python versions
  --artifact [ARTIFACT ...], -a [ARTIFACT ...]
                        Manually provide the artifact URLs to download
  --output OUTPUT, -o OUTPUT
                        The output path
  --timeout TIMEOUT, -t TIMEOUT
                        Timeout hours

Ex.
python download_wheels.py -s https://tensorflow-ci.intel.com/ -j tf-rel-win -v 2.12.0rc0 -p 311 -p 310 -p 39 -p 38 -o C:/Users/Prody/Downloads
python download.py -s https://tensorflow-ci.intel.com/ -j tf-rel-win -v 2.12.0rc0 -o C:/Users/Prody/Downloads -a https://tensorflow-ci.intel.com/job/tf-rel-win/74/
"""
import argparse
import doctest
import sys
import os
import re

from pathlib import Path
from urllib import request, parse
from typing import Optional, Tuple
from collections.abc import Iterable

import jenkins

from bs4 import BeautifulSoup


class ToPackagePyVersion(argparse.Action):
    def __init__(self, option_strings, dest, *args, **kwargs):
        super().__init__(option_strings, dest, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=''):
        if not values:
            raise ValueError(f'{option_string} is required!')
        attr: list = getattr(namespace, self.dest, [])
        attr.extend([f"cp{v}" for v in values])
        setattr(namespace, self.dest, attr)


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object
    >>> p = parse_args(['-s', 'j.com', '-j', 'tf', '-v', '2.0rc', '-p', '3', '-p', '4', '-a', '/a', '-a', '/b'])
    >>> p.server
    'j.com'
    >>> p.job
    'tf'
    >>> p.version
    '2.0rc'
    >>> p.py_version
    ['cp3', 'cp4']
    >>> p.artifact
    ['/a', '/b']
    >>> p.timeout
    1
    >>> p.output
    WindowsPath('.')
    """
    p = argparse.ArgumentParser(
        description="Wait for a specific python package version from https://pypi.org/rss/project/package/releases.xml")
    p.add_argument("--server",
                   "-s",
                   type=str,
                   required=True,
                   help="The Jenkins server url")
    p.add_argument("--job",
                   "-j",
                   type=str,
                   required=True,
                   help="The job url to search")
    p.add_argument("--version",
                   "-v",
                   type=str,
                   required=True,
                   help="The desire package version to search from the job history")
    p.add_argument("--py_version",
                   "-p",
                   action=ToPackagePyVersion,
                   type=str,
                   default=[],
                   required=False,
                   nargs="*",
                   help="The required package Python versions")
    p.add_argument("--artifact",
                   "-a",
                   action="extend",
                   type=str,
                   default=[],
                   required=False,
                   nargs="*",
                   help="Manually provide the artifact URLs to download")
    p.add_argument("--output",
                   "-o",
                   type=Path,
                   required=False,
                   default='.',
                   help="The output path")
    p.add_argument("--timeout",
                   "-t",
                   type=int,
                   default=1,
                   required=False,
                   help="Timeout hours")

    if isinstance(test_args, str):
        return p.parse_args(test_args.split())
    elif isinstance(test_args, list):
        return p.parse_args(test_args)
    else:
        return p.parse_args()


def exit_error():
    sys.exit(-1)


def print_error(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def get_versions(name: str) -> Tuple[str, str]:
    _, version, py_version, _, _ = name.split('-')
    return version, py_version


def get_build_info(url: str) -> Tuple[str, str]:
    parts = url.split('/')
    index = parts.index('job') + 1
    name, build = parts[index: index + 2]
    return name, build


def get_wheel_url(url: str) -> str:
    r = request.Request(url, method='GET')
    rp = request.urlopen(r)
    soup = BeautifulSoup(rp.read(), 'html.parser')
    re_wheel = re.compile('.*\.whl')
    links = soup.find_all('a', attrs={'href': re_wheel}, string=re_wheel)
    if links:
        anchor = links[0]
        return parse.urljoin(url, anchor['href'])
    else:
        return ''


def download_artifacts(urls: Iterable[str], output: Path, version: str):
    if not output.is_dir():
        os.mkdir(output)

    for url in urls:
        wheel_url = get_wheel_url(url)
        if not wheel_url:
            print_error(f'Error: Cannot find wheel URL from build {url}')
            exit_error()

        print(f'Downloading {wheel_url} from build {url}')
        r = request.Request(wheel_url, method='GET')
        rp = request.urlopen(r)
        name = wheel_url.split('/')[-1]
        wheel_version, _ = get_versions(name)
        if wheel_version != version:
            print_error(f'Error: The wheel version {wheel_version} is not match the required {version}!')
            exit_error()

        with open(output.joinpath(name), 'wb') as f:
            f.write(rp.read())


def get_artifact_links(server: str, job: str, version: str, py_versions: Iterable[str], timeout_hour: int) -> Iterable[str]:
    """
    Get artifact links for the job from the server

    Args:
        server (str): Jenkins server URL
        job (str): Jenkins job name
        version (str): The package version will be looking for from the job history
        py_versions (Iterable[str]): The required python package versions
        timeout_hour (int): Timeout hour

    Returns:
        Iterable[str]: The found artifact URLs

    Reference:
        https://python-jenkins.readthedocs.io/en/latest/examples.html#example-1-get-version-of-jenkins
        https://python-jenkins.readthedocs.io/en/latest/api.html
    """
    server = jenkins.Jenkins(server)
    if not server.wait_for_normal_op(timeout_hour * 3600):
        return []

    artifacts = []
    py_versions = list(py_versions)
    job_info = server.get_job_info(job)
    for build in job_info['builds']:
        build_number = build['number']
        build_info = server.get_build_info(job, build_number)
        if build_info['result'] == 'SUCCESS':
            for a in build_info['artifacts']:
                name: str = a['fileName']
                if name and name.endswith('.whl'):
                    wheel_version, py_version = get_versions(name)
                    if wheel_version == version and py_version in py_versions:
                        url = build_info['url']
                        print(f'Found {job} version {version} artifact for Python {py_version} {url}')
                        artifacts.append(url)
                        py_versions.remove(py_version)

                        if not py_versions:
                            return artifacts
                        else:
                            break
    if py_versions:
        print_error(f"Error can't find all the required package version {version} for Python {py_versions}")
        exit_error()
    else:
        return artifacts


def main(server: str, job: str, version: str, py_versions: Iterable[int], artifacts: Iterable[str], output: [str, Path], timeout_hour: int):
    if not artifacts:
        artifacts = get_artifact_links(
            server, job, version, py_versions, timeout_hour)
    elif artifacts and py_versions:
        print('Warning: The artifact parameter exist. The py_versions parameter will be ignore')

    download_artifacts(artifacts, output, version)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for wait_for_release.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        args = parse_args()
        main(args.server,
             args.job,
             args.version,
             args.py_version,
             args.artifact,
             args.output,
             args.timeout)
