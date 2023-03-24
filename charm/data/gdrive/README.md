# Google File Data Storage Notes
## Uploading files
You can upload any file format to Google Drive using `upload.py`. You can also create a folder where the data should be uploaded and optionally specify which folder it should live in by calling `upload.py::create_folder` and specifying the parent directory where the folder should be created.

**TODO:**
- Will need to gather up all files to be uploaded
- Create intermediate folders
- Record file IDs along the way so they can be linked in the metadata sheet
- Determine if we can upload in parallel
- Keep track of what has been uploaded and figure out how `resumable` works
- Time upload speed

## Downloading files
Files can be downloaded using `download.py` by specifying a file ID.

**TODO:**
- Determine if we can download in parallel
- Time download speed

## Listing files
**TODO:**
- Implement code to list files