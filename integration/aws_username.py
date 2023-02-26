"""This patches a silly AWS bug by adding the expected User Name column to the 
rootkey.csv file.

Examples:
    $ python aws_username.py \
        --key-filepath ~/.ssh/user1_accessKeys.csv \
        --user-name user1
"""
import argparse
import pandas as pd


def main(args):
    df = pd.read_csv(args.key_filepath)
    if 'User Name' not in df.columns:
        df.insert(loc=0, column='User Name', value=args.user_name)
        df.to_csv(args.key_filepath, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key-filepath',
                        help='The path the AWS provided rootkey.csv file.',
                        required=True)
    parser.add_argument('--user-name',
                        help='The user name to specify in the User Name column.',
                        required=True)
    args = parser.parse_args()
    main(args)