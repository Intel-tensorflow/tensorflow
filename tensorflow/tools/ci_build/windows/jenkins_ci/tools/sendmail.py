#!/usr/bin/env python3
"""
The Email notification helper script

Ex:
usage: sendmail.py [-h] [--from_address FROM_ADDRESS] --to_addresses TO_ADDRESSES [TO_ADDRESSES ...]
                   [--subject SUBJECT] [--content CONTENT] [--key KEY]

The Email notification helper script

optional arguments:
  -h, --help            show this help message and exit
  --from_address FROM_ADDRESS, -f FROM_ADDRESS
                        The sender address
  --to_addresses TO_ADDRESSES [TO_ADDRESSES ...], -t TO_ADDRESSES [TO_ADDRESSES ...]
                        The Email to address. Can be supplied multiple times.
  --subject SUBJECT, -s SUBJECT
                        The subject of the Email.
  --content CONTENT, -c CONTENT
                        The Email content string.
  --key KEY, -k KEY     The private key to decrypt the credentials.

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

creds_content = b'gAAAAABjqvE1Zj429Nqq4XlPN4Qiko8nmcetSuL68yjN5gj02_rW-tjglAUM59YedkPA_bwkB-Q_tBcMDfKWI3oKPVozbuJGUD4yQt1c5OeU6K-mR-1pXhMNWeF1yr8n-l6Ow_wYEXHxvPFUpd2HAF9f6e-MeTCpA10zd9yqOhmh2z1gLWeHbyRAtTT8FkBP01rnTddp26D7AdMGHBPhublQ9nzY5GbNZe488sS_Gby0AhuhwNi9S0WoplArW-fj37KYUM8ASEcdqboZIR4pCm37cXj7Jn5BvfXpON7l3SaiOco0wWnxeajtehi-JtxXHpz3OHsVfjl0fgQ2M1G9sUnoSXaKnlwie9wkCjyz1znamxaBMHWQ0NTjlFb7BGHT2BEogugNOokB9GCIgchuuo3vvBz7LHkj8IG5LzNfC7I7nv3R2wsmQ8IWUnf1hNASi5g12vxurF-sk4KBA_R0CtpMBmMxSDE_EM6mQxfrsEVSqtEUMq2u1SzHAxVfpYzCYO7MA6fMj72nTT3alz64USg5pp-AXRb19GouVc8mQmNQBZGDe_ft98S-nywlYVZ7JuDpv9_UJ3lOGzyr4OGTX_1ByBApgCwnFQrk-ok0lLr-szSBmR4OKCo='
token_content = b'gAAAAABkQhz_c2OjDcCD5ATRC_0wXxASPeHJRumjeMa2LDDEpCOn_jV_pa2fwiqFtHDU3u0D_X8HYpLj2D-jERERGn7YpMbYhR-u06I5x2Z6xLJmRc-NGes4uNa9SFpP3ql_yQjsoQp7GNrMA1XY_hux8OCMYJPx63IPTDvTT4-KR-KewE61APAGGB8vQ1ZYbGt3AN1qvdEE41vGMhxF1tTsfcqLf3ShxTB3XCu2Wr3CoV6n_i53w9d2RyoCN2G4YV9G-AMqYQFYDnledbxmmyZ0RgQcKM01G0x5tmJB8kTfajEEcoW86WKuS0ESBjZqefCesVJ4Z91r_qc4ZBegXMNIlqwuWRUAWfDIQamj7H04Fc3y3kEVGurTFZQoWLi7QWqGJhlrl67IaakcWDWxSh8ranUAIkYLjgee8knOIVd11Hgj6wGzt9X5Jdmddy7HePJLiMD-C3-kEDy8zyolvEax5IrXm40x0jY2S1BOlge7wnPsgXlToKNPQPCOMQz91v2JLU3VOXd5_X_ubLv05UfZmmTOjIireHjVYM6qAv3on7N1-3EUlDSQuOjJ3IzYbQXN2JYY_HhnTRIC9FF56vVPgDapZSTQIp4lBkp3X7EJk4Zv_9h4-qKwHLS6GaTObtg1dzavLDioRL1gLTYvvqcZVVhw78Vs6cS8CsytzCSskw9r4DAq1Yq-McfqzLFW5f_Vypl2zCgJSR1UlNw4Kkb2yfopU_mHzUffeMZpoDQ8AGbshPUxMb1CuH2Zqtaj-RbngF_stHrTamNwyzqObEOlT21zxt__lZkCjL0xmQOKm56JaVjzRZC-HDrkpS7D3pi9PcFjvhd1506SEyL6dNI0qtHgMRkdsj1B9LABURysSvgG8kYm3PglmsUmDv775YodxuRfXFWhnZU5maB_85e33gTRHIkXB6EYkzAFJjHHhqZ5PDUK0N0jgYydDVyLcG8ykVAQBHWt-fr4oGO39ccL1KQJObUqdhbqUfaFUMoKxK_6WsPumoqC2bEnSI-5RBpTl0e0Ek34elVinM2dgl-obLSc8JbYew=='


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
    else:
        print(f"Create Email credential failed due to token file is not exist {token_path}")
        sys.exit(1)

    # If there are no (valid) credentials available, let the user log in.
    if creds:
        try:
            creds.refresh(Request())
        except Exception:
            print('Refresh credential failed due to token expired')
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

    if not creds.valid:
        print(f"Email credential {creds_path} is not valid. Send Email failed!")
        sys.exit(1)

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
