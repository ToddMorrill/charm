LDC video download tool


REQUIREMENTS:

- Docker + internet connection (https://docs.docker.com/get-docker/)

USAGE:

The downloader process, including Docker image build, can be run by executing get_urls.sh and passing the
URL list file as a single argument:

./get_urls.sh <URL_list_file> 

The URL list file must be located in the same directory as get_urls.sh and
must be a tab-delimited file with three columns:

source_uid	file_uid	URL

Downloaded and wrapped videos will be saved to the 'out' directory.

VERSION HISTORY:
20221005	V1.0
