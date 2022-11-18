#!/usr/bin/env python3
"""
Find the first bad commit base on git bisect search.

It is depends on a third party jenkins library ujenkins. Please install it by
$ pip3 install ujenkins

Ex:
find_first_invalid_commit.py [-h] [--good GOOD] [--bad BAD] [--branch BRANCH]
                                    --cmds [CMD ...] [--oneline] [--filters FILTER]
Switch to the branch and run git bisect search for the repository

Unit test:
Execute the script without any arguments.
"""
import argparse
import doctest
import sys
import os
import re

from typing import Optional, Union, Sequence
from pathlib import Path
from subprocess import getstatusoutput
from collections import namedtuple
from urllib.parse import urlparse, ParseResult

from ujenkins import JenkinsClient


JenkinsInfo = namedtuple("JenkinsInfo", ["server", "job", "build"])
EmptyJenkinsInfo = JenkinsInfo("", "", "")


class GitCmdError(Exception):
    def __init__(self, message, *args, **kwargs):
        super().__init__(message, *args, **kwargs)


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args(['repository', '-g', '10ae', '-b', '53e1', '-r',
    ... 'abc', '-f', 'Error message', '-f', 'Failure', '-c', 'git status',
    ... '-c', 'bazel build', '-j', 'https://tensorflow-ci.intel.com/Job/tf-test-win2/1962'])
    >>> p.repository
    WindowsPath('repository')
    >>> p.good
    '10ae'
    >>> p.bad
    '53e1'
    >>> p.filters
    ['Error message', 'Failure']
    >>> p.branch
    'abc'
    >>> p.cmds
    ['git status', 'bazel build']
    >>> p.oneline
    False
    >>> p.job
    'https://tensorflow-ci.intel.com/Job/tf-test-win2/1962'
    """
    p = argparse.ArgumentParser(description="Automatic find the error commits base on git bisect command. The script is depend on an third party library ujenkins.")
    p.add_argument("repository",
                   type=Path,
                   help="Repository path")
    p.add_argument("--good",
                   "-g",
                   type=str,
                   required=False,
                   help="A good commit id")
    p.add_argument("--bad",
                   "-b",
                   type=str,
                   required=False,
                   help="A bad commit id")
    p.add_argument("--branch",
                   "-r",
                   type=str,
                   required=False,
                   default="",
                   help="Search from a branch")
    p.add_argument("--oneline",
                   "-o",
                   action="store_true",
                   required=False,
                   default=False,
                   help="Simply the output to one line commit id")
    p.add_argument("--cmds",
                   "-c",
                   nargs="+",
                   action="extend",
                   required="True",
                   help="Build command for validation")
    p.add_argument("--job",
                   "-j",
                   type=str,
                   required=False,
                   help="A jenkins job url. ex. https://tensorflow-ci.intel.com/Job/tf-test-win2/1962. Without build id the lastBuild will be selected")
    p.add_argument("--filters",
                   "-f",
                   nargs="*",
                   action="extend",
                   required="True",
                   help="The filters will be used compare with the output to identify the target commit. Support multiple filters ex. -f filter1 -f filter2")

    if isinstance(test_args, str):
        return p.parse_args(test_args.split())
    elif isinstance(test_args, list):
        return p.parse_args(test_args)
    else:
        return p.parse_args()


def _git_cmd(cmd: str, is_ignore_error: bool = False):
    result, output = getstatusoutput(cmd)
    if result and not is_ignore_error:
        raise GitCmdError(f"Execute git cmd `{cmd}` failed!")

    return output


def _validate_commit(commit: str):
    if not re.match("[0-9a-zA-Z]+", commit):
        raise ValueError(f"Commit {commit} is not valid!")


def _update_repository(path: Path, branch: Optional[Union[str, None]] = None):
    if path.exists():
        os.chdir(path)
    else:
        raise ValueError(f"The path {path} is not exist!")

    _git_cmd("git bisect reset", True)
    _git_cmd("git reset --hard", True)

    if branch:
        _git_cmd(f"git checkout {branch}")

    _git_cmd("git pull", True)


def _start(good: str, bad: str):
    _git_cmd("git bisect start")
    _mark_good(good)
    _mark_bad(bad)


def _mark_good(commit: str = ""):
    _git_cmd(f"git bisect good {commit}")


def _mark_bad(commit: str = ""):
    _git_cmd(f"git bisect bad {commit}")


def _next() -> int:
    output = _git_cmd("git bisect next")
    return _remain_steps(output)


def _get_bad_commit(git_cmd=_git_cmd) -> str:
    """
    Get bad commit id

    Returns:
        str: the first bad commit id

    >>> s = ["234d6df865d069f1602fae68aa5c58679544f910 is the first bad commit",
    ... "commit 234d6df865d069f1602fae68aa5c58679544f910",
    ... "Author: ..."]
    >>> from unittest.mock import MagicMock
    >>> mgit_cmd = MagicMock(return_value="".join(s))
    >>> _get_bad_commit(mgit_cmd)
    '234d6df865d069f1602fae68aa5c58679544f910'
    """
    output = git_cmd("git bisect next")
    m = re.match("([\w\d]+)\s+is the first bad commit", output)
    if m:
        return m.group(1)
    else:
        return ""


def _checkout(commit: str):
    _git_cmd(f"git checkout {commit}")


def _complete():
    _git_cmd("git bisect reset")


def _remain_steps(output: str) -> int:
    """
    Get the remain git bisect steps from the output

    Args:
        output (str): the git bisect output

    Returns:
        int: the remain step number
    >>> _remain_steps('Bisecting: 0 revisions left to test after this (roughly 1 step)')
    1
    """
    lines = output.splitlines()
    if lines:
        m = re.match(".*\(roughly (\d+) step", lines[0])
        if m:
            return int(m.group(1)) + 1

    return 0


def _cmd(cmd: str = "", filters: Sequence[str] = []) -> int:
    """
    Execute the command and filter the output base on the filters

    Args:
        cmd (str, optional): a command line will be executed.
        filters (Sequence[str], optional): the filters will be used to check the output. Defaults to [].

    Returns:
        int: 0 or 1 to indicate the command execute successful. If the output
        contains the filter then 1 will be returned.

    >>> _cmd("echo hello world", ["hello"])
    1
    """
    if cmd:
        result, output = getstatusoutput(cmd)
        if filters and output and any(filter in output for filter in filters):
            return 1
        else:
            return result
    else:
        return 0


def _get_changesets_from_jenkins(jenkins_server: str, job_name: str, build: str, ) -> Sequence[str]:
    """
    Get the changesets information from a given jenkins job for a specific build

    Args:
        job_name (str): The jenkins job name
        build (str, optional): build number such as 1955 etc. Defaults to "lastBuild".
        jenkins_server (str, optional): The jenkins server name. Defaults to "https://tensorflow-ci.intel.com/".

    Returns:
        Sequence[str]: Return all the found changes from first to last
    """
    try:
        c = JenkinsClient(jenkins_server)
        sets = c.builds.get_info(job_name, build)['changeSets']
        return tuple(item['commitId'] for item in [s['items'] for s in sets][0])
    except Exception as e:
        raise ValueError(
            f"Cannot get the changesets from the jenkins server {jenkins_server} {job_name} {build} with error:{e}!")


def _parse_jenkins_url(url: str) -> JenkinsInfo:
    """
    Parser jenkins information from a job url.

    Args:
        url (str): jenkins job url

    Returns:
        JenkinsInfo: a named tuple with fields server, job, build

    >>> _parse_jenkins_url("https://tensorflow-ci.intel.com/Job/tf-test-win2/1962/console")
    JenkinsInfo(server='https://tensorflow-ci.intel.com', job='tf-test-win2', build='1962')
    >>> _parse_jenkins_url("https://tensorflow-ci.intel.com/Job/tf-test-win2")
    JenkinsInfo(server='https://tensorflow-ci.intel.com', job='tf-test-win2', build='')
    >>> _parse_jenkins_url("http://abc.com")
    JenkinsInfo(server='', job='', build='')
    """
    global EmptyJenkinsInfo

    if not url:
        return EmptyJenkinsInfo

    result: ParseResult = urlparse(url.lower())
    if result:
        paths = result.path.split("/")
        scheme = result.scheme or "https"
        if "job" in paths:
            job_name_index = paths.index('job') + 1
            build = ""
            paths = paths[job_name_index:]
            if len(paths) > 1:
                job_name = paths[0]
                build = paths[1]
            else:
                job_name = paths[0]

            return JenkinsInfo(f"{scheme}://{result.netloc}", job_name, build)

    return EmptyJenkinsInfo


def main(repository: Union[str, Path],
         good: str,
         bad: str,
         cmds: Sequence[str],
         branch: Optional[str] = "",
         filters: Optional[Sequence[str]] = [],
         job: str = "",
         oneline: bool = False):
    """
    The entry for the find_first_invalid_commit.py

    Args:
        repository (Union[str, Path]): Git repository path
        good (str): The good commit id
        bad (str): The bad commit id
        cmds (Sequence[str]): The commands will be used to validate the result
        branch (str, optional): The branch name. Defaults to "".
        filters (Sequence[str], optional): The filters message string which will be used to filter the cmd output. If found the commit will be mark as bad. Defaults to "".
        job (str, optional): The jenkins job url, it can contain the build id.
        oneline (bool, optional): Output will only contain the found commit id. Defaults to False.
    """
    if job:
        jenkinsInfo: JenkinsInfo = _parse_jenkins_url(job)
        if jenkinsInfo == EmptyJenkinsInfo:
            raise ValueError(f"ERROR cannot find jenkins job information from the job path {job}")

        build = jenkinsInfo.build or "lastBuild"
        changesets: Sequence[str] = _get_changesets_from_jenkins(
            jenkinsInfo.server, jenkinsInfo.job, build)
        good, bad = changesets[0], changesets[-1]

    _validate_commit(good)
    _validate_commit(bad)

    if not isinstance(repository, Path):
        repository = Path(repository)

    _update_repository(repository, branch)
    _start(good, bad)

    while _next():
        if any(_cmd(c, filters) for c in cmds):
            _mark_bad()
        else:
            _mark_good()

    if oneline:
        print(_get_bad_commit())
    else:
        print(f"The first suspicious commit is {_get_bad_commit()}")

    _complete()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for find_first_invalid_commit.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        args = parse_args()
        main(args.repository,
             args.good,
             args.bad,
             args.cmds,
             args.branch,
             args.filters,
             args.job,
             args.oneline)
