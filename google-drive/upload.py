from __future__ import print_function
import os
import argparse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

import utils


def create_folder(parents=[]):
    """ Create a folder and prints the folder ID
    Returns : Folder Id

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    creds = utils.get_creds()

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {
            'name': 'test-directory',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': parents
        }

        # pylint: disable=maybe-no-member
        file = service.files().create(body=file_metadata).execute()
        print(F'Folder ID: "{file.get("id")}".')
        return file.get('id')

    except HttpError as error:
        print(F'An error occurred: {error}')
        return None


def upload_basic(filepath, parents=[]):
    """Insert new file.
    Returns : Id's of the file uploaded

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    creds = utils.get_creds()

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        path, filename = os.path.split(filepath)
        file_metadata = {'name': filename, 'parents': parents}
        media = MediaFileUpload(filepath, mimetype='image/png')
        # pylint: disable=maybe-no-member
        file = service.files().create(body=file_metadata,
                                      media_body=media).execute()
        print(F'File ID: {file.get("id")}')

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    return file.get('id')


def main(args):
    # TODO: check if there is an upload in progress and resume progress using
    # the meta_file. Create meta_file if it doesn't exist.
    # TODO: list out all files and compute diff against meta_file
    # TODO: upload files and respect folder structure
    
    folder_id = create_folder(['1zbGQcYl63sREkVm0v-JkhEi6gbNr_tLj'])
    file_id = upload_basic('./ccu-619.png', parents=[folder_id])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-file',
                        help=('The metadata file containing information about '
                        'the upload.'))
    parser.add_argument('--directory',
                        help=('The parent directory containing '
                              'subdirectories and files to be uploaded.'))
    args = parser.parse_args()
    main(args)
