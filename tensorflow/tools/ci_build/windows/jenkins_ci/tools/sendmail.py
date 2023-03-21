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
token_content = b'gAAAAABkGUU4dS-BF4W0R8m-aopWxKaVl9biTsDVrVzB4Wv5adM5Y7lmLdJ3TmLAP5yn_hlwcjgku5jOLypdMGu_QcT-C1oCcGDbU136RoZ3o767O4ROHGCYk99MOEQcpgrwmoQOpHMry1DUDQdr84iYmShQTkh2-ljJLfpjVP3sW30AG_Jrh_vuVu_niSwcku4uA1iJrdZPZmtTKTvjD_YOpG_i57JkXEwZQUA91ECCxXWbP0CF5ucn2AEJmixcXDSZMJzoBG6zJpl6WPF9R12jS0B7RgX3_kv-sOLdLF4DE-d3uOxScbzoHrXS7UygSFYmuKg5EHCzxJuj_iRsQXiC6OefzDsvbx1feL9AN52RFivZZc7OZsqL4q4f6wcCR0t89xss6fCvVhTbEXLMuGfDbPqLr5N6n1gNjusqEB_TgZGoogKFhOLUYDnNcq6OVBsDLMtA7FLc48zl8nZpDQnZifTvRlKYpfQovvP0Bw0-PhpuNhP9X8YNdUAWaBu2QlaMZatUhRoQS72oy7YVFjnmkDJ5Ipd11gsahtUvS0TM53V8X-fzp0z_FeP3kpSUPaxAY3bKr8vg7J7PUMHzAVqObrkxc07mTsZeQXbG343mwks_QUi1dHJnzKrlYwlOfAckYQXz9sZSiyN9amOv4xTY5rR6v5BihL0uwP9nsUiYXxQETaGuEhXd26g-LbOdd1i5gSEEKWoW9sQ2-pw73GMWMluVHohIAKHmVV03F85004OBxQ5KuDtNVrG3tnE00ks9PXkiXTeN6N1HhXEwTxJMwPXzJiExuw25iFV8Ty0Bx2VAYVRsxp5V30hFW460LQVsfqnGPXWip-OFshrPWQnBt9y5zlurmv2ypI2xQYMOVlESdMEpJwR9tfuXGgItxFVHiaxr9-_s-EOoCEkZ9bsLk9VTPlcgt4PZ7xkFd2rv9QGf4yyzadHE9GOy__bRnMkTS6izvHqUGy4uMYhVx3pZryeQornYho5gNlBODTrbZhJHPKlMEkbxzi2WSuMrLkqMcSzUuxIFc1mEUc2kWIAL4XB-Srw49Q=='


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
