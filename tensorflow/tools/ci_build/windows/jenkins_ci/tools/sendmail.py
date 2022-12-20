#!/usr/bin/env python3
"""
The Email notification helper script

Ex:
find_first_invalid_commit.py [-h] [--good GOOD] [--bad BAD] [--branch BRANCH]
                                    --cmds [CMD ...] [--oneline] [--filters FILTER]
Switch to the branch and run git bisect search for the repository

Unit test:
Execute the script without any arguments.

Requirements:
$ pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib cryptography

Note:
The helper script requires additional two steps:
1. Configure the gmail account and google cloud application
https://developers.google.com/gmail/api/quickstart/python?hl=en

2. Provide the decryption key which generated with the cryptography library
"""
import base64
import os.path
import argparse
import doctest
import sys
import os
import shutil

from tempfile import mkdtemp
from typing import Optional, List
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from cryptography.fernet import Fernet

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://mail.google.com/',
          'https://www.googleapis.com/auth/gmail.send']
creds_root = mkdtemp()
creds_path = os.path.join(creds_root, 'credential.json')
token_path = os.path.join(creds_root, 'token.json')

creds_content = b'gAAAAABjodNrilAjbBeFHAs6E3qMS5PGfTCIycG2pfwKF_SKtimYSyDn-9iuV3MmMZZzWu-BOwIcO-0xE09xTzFdOp2sCzPB0iv_lvBwmSG7bA48glCE9dTb1pjEObi3Pz9e-aBTDhVikuWBUVxaZJ12otYuzRTR-05m6DzP0roD9gfEmvzm_wvIErkArl7hktVcLb2847-1GUBeT9Vh7rWpbnF4DgfMq1RiT9ElimpQX-FZshLk5AVNnQ_vmWCgdTq_aJfVRaIDx0l9wFEXhksWh1pmYHge-PRR3INLye8bvEdqFNjX6lV9reVd33WrmQO_Jb98bApcDsDi86UNx3ZVyWznR0fSZ2m3Y0450rQHN9D0r4QZn0MpQF3tImmQw2oQUmNEqRv_qoJgnsxKFYbdsUtGtDmJ3dLrTJu8KvpMQ9mIamJwGsv5v81t6eZL0ULZyXGZDStehCDBJqD_SXZArLZZLSCutA4dapgzFGJg7ik0BxwNlB6e2QKoVT5S9dr6l6V-Do6yjbdwXukaJwF94upvA4jLlsNmG4-wCqDkjoQ266SAOw_WwUD2_kKi3T3AQBJRanG4zDXjr-KbLGf4zp50ZpKFqk-gO9K6F2arzGuA8XHxtyw='
token_content = b'gAAAAABjodNPC9UzM2DPB0Q_VSFNwS9JOMyyw9tok3bAXs8RYdLDauaHctByovVLUGVSHe8bzVuM3gLNM6_bDYy7xTZZ6DUjlQ04_55_bcS5uiv3xpKIKqPvQxQypoVfqO6vvTXvC_OnwW-rFYxmbAG0vDW8QFYF2jgKee-zjfPh1sA3KkCsirg5vKCZbWueHMyQcEAak_h7ytzXXpiZEBPndJkUgYm3GbbFbfLJsjto6m66TVFFfct--3GfprgO7MhXDb0vuOIbTTnQEiRS-Km-8K9gVaiwbyD9Ro8iE5zYgtGL4MBeK5nXcjyPeGu-lbceSKFkp81vydvxZrZNyflGORnxcgX71amg-jRxH3cFv5U4LBdGX3wH29qhyXbIodgncdSJ36seGVivGVNgl1ZXHoG1QQdbQPHxWvxTOWyHDxEyvTcrqhOGvwSCqLdLsyErKTlTZW8lGBWkEWwP4nE2mUKgsAFwwL-fZkC_RHhbm0qs3fMI-DF0IeUdefu9cwSb53WXr4ZNin9gRlt_COfq1kwcxKY0uBOemrfKURiYYGUWpRtAwtoVbUjoqlp5yWkxyiPY8Hx929g1acFKK905uFshK9kVhk_ZFVIfkfm6Cfgt8OM_icTtmmzLmT5asmepMBVG99zIj96S0i3sbVZJqyAcH8BsYqmSEBECFaJAQCFy45FCWxi74JskJsnTBj2mDcW0SdPGV_QaWxItSS9FqlvFHUpKbpfLaTXbRBqlbmPWJhkkcOZft-zfxHHWGU3_YSAM6rUiDUOn25H8jN0-hkh7WPzvaHEOY312HDXcuMVArEy6tFJMFlQu7a3Tb4MsdpY4BMHvk_zTBtL-1rzR8FmJD7TAQLVkyEul8Q7CNZ6ZRWXk_bjZBCbotx4R6U16vlJBoFH5TDwPdMiDlPsHXbPA8hBKdYNn2eu5eL7xqYBT5JrPeV3rPVP0wuO59bpoktKGtiH5XgeDUksne5idR2uEtCgsED5nPW2mieZx46VfxJz0xzujm3ccXYYuJh2a2Evzx4NypjZxfPqM-j5zYfK3dHoOvQ=='


class ToBytes(argparse.Action):
    def __init__(self, option_strings, dest, *args, **kwargs):
        super().__init__(option_strings, dest, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=''):
        if not values:
            raise ValueError(f'{option_string} is required!')
        setattr(namespace, self.dest, bytes(values, 'utf-8'))


def parse_args(test_args: Optional[str] = None) -> argparse.ArgumentParser:
    """Parser the command line arguments

    Returns:
        argparse.ArgumentParser: return the parser object

    >>> p = parse_args(['-t', 't0', '-t',
    ... 't1', '-s', 'subject', '-c', 'content', '-k', 'abc'])
    >>> p.from_address
    'intel.tensorflow.robot@gmail.com'
    >>> p.to_addresses
    ['t0', 't1']
    >>> p.subject
    'subject'
    >>> p.content
    'content'
    >>> p.key
    b'abc'
    """
    p = argparse.ArgumentParser(
        description="The Email notification helper script")
    p.add_argument("--from_address",
                   "-f",
                   type=str,
                   default="intel.tensorflow.robot@gmail.com",
                   required=False,
                   help="The sender address")
    p.add_argument("--to_addresses",
                   "-t",
                   type=str,
                   nargs="+",
                   action="extend",
                   required=True,
                   help="The Email to address. Can be supplied multiple times.")
    p.add_argument("--subject",
                   "-s",
                   type=str,
                   required=False,
                   default="",
                   help="The subject of the Email.")
    p.add_argument("--content",
                   "-c",
                   type=str,
                   required=False,
                   default="",
                   help="The Email content string.")
    p.add_argument("--key",
                   "-k",
                   action=ToBytes,
                   help="The private key to decrypt the credentials.")

    if isinstance(test_args, str):
        return p.parse_args(test_args.split())
    elif isinstance(test_args, list):
        return p.parse_args(test_args)
    else:
        return p.parse_args()


def decrypt_credential(key: bytes, encrypted: str) -> bytes:
    """
    Decrypt the credential with the private key.

    >>> k = Fernet.generate_key()
    >>> f = Fernet(k)
    >>> s = f.encrypt(b'data')
    >>> decrypt_credential(k, s)
    b'data'
    """
    f = Fernet(key)
    return f.decrypt(encrypted)


def dump_credential(key: bytes, to_path: str, credential: bytes):
    if not os.path.exists(to_path):
        with open(to_path, 'wb') as c:
            c.write(decrypt_credential(key, credential))


def dump_credentials(key: bytes):
    global creds_path
    global token_path
    global creds_content
    global token_content

    dump_credential(key, creds_path, creds_content)
    dump_credential(key, token_path, token_content)


def clean():
    global creds_root
    shutil.rmtree(creds_root)


def sendmail(from_address: str, to_addresses: List[str], subject: str, content: str):
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    global creds_path
    global token_path

    creds = None
    to_addresses = ';'.join(to_addresses)
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    try:
        # create gmail api client
        service = build('gmail', 'v1', credentials=creds)

        message = EmailMessage()
        message.set_content(content)
        message['To'] = to_addresses
        message['From'] = from_address
        message['Subject'] = subject

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {'raw': encoded_message}
        service.users()                                \
               .messages()                             \
               .send(userId="me", body=create_message) \
               .execute()
        print(F'Send Email to: {to_addresses}')

    except HttpError as error:
        print(F'An error occurred: {error}')


def main(from_address: str, to_addresses: List[str], subject: str, content: str, key: bytes):
    dump_credentials(key)
    sendmail(from_address, to_addresses, subject, content)
    clean()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("*** Run doctest for sendmail.py! ***")
        doctest.testmod(optionflags=doctest.ELLIPSIS |
                        doctest.IGNORE_EXCEPTION_DETAIL)
    else:
        args = parse_args()
        main(args.from_address,
             args.to_addresses,
             args.subject,
             args.content,
             args.key)
