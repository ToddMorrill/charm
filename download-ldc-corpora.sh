#!/bin/bash

################################################################################
#                                                                              #
# Script to download corpora from the Linguistic Data Consortium               #
#                                                                              #
# Original author: Lane Schwartz                                               #
# Additional contributors: None                                                #
#                                                                              #
#                                                                              #
# Change log:                                                                  #
#                                                                              #
#  * 2016-03-05 Lane Schwartz                                                  #
#    Verify that downloaded files are tgz, not HTML.                           #
#    If HTML, exit the script, because file was not correctly downloaded       #
#                                                                              #
#  * 2016-03-05 Lane Schwartz                                                  #
#    Download a richer set of information into the corpora.tsv file            #
#    and record that information in the names of the downloaded files          #
#                                                                              #
#  * Created on 2016-03-04 by Lane Schwartz                                    #
#    Based on Python code by Jonathan May:   https://github.com/jonmay/ldcdl   #
#                                                                              #
#                                                                              #
# This program is free software: you can redistribute it and/or modify         #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
# This program is distributed in the hope that it will be useful,              #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU Lesser General Public License for more details.                         #
#                                                                              #
# You should have received a copy of the GNU General Public License            #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#                                                                              #
################################################################################





################################################################################
#                                                                              #
#                        Bash functions and variables                          #
#                                                                              #
################################################################################

# Define a bash function for logging messages with a timestamp
log() {
    TIMESTAMP=$(date +"%Y-%M-%d %H:%M:%S UTC%:::z")
    printf "%-27s %-65s %-3s" "${TIMESTAMP}" "${1}..." 1>&2 ;
}

# Define required URL variables
BASE_URL="https://catalog.ldc.upenn.edu"
LOGIN_URL="https://catalog.ldc.upenn.edu/login"
DOWNLOADS_URL="https://catalog.ldc.upenn.edu/organization/downloads"

DOWNLOAD_FILE="corpora.tsv"


EMAIL=$1
PASSWORD=$2


################################################################################
#                                                                              #
#                         Access list of LDC corpora                           #
#                                                                              #
################################################################################

log "Accessing list of your LDC corpora"


curl -k -L --user "${EMAIL}:${PASSWORD}" "${DOWNLOADS_URL}" > ${DOWNLOAD_FILE} 2> /dev/null


################################################################################
#                                                                              #
#                       Provide user with status update                        #
#                                                                              #
################################################################################

echo
log "A list of the LDC corpora associated with your account has been saved to ${DOWNLOAD_FILE}"
echo

if [[ "$*" == "" ]]; then

    log "To download any of these corpora, re-run this script, providing the LDC corpus IDs as command line arguments"
    echo

fi

################################################################################
#                                                                              #
#                            Download LDC corpora                              #
#                                                                              #
################################################################################

for LDC_CORPUS in "${@:3}" ; do

    echo

    # this grep is a little hacky, might break but if it does
    # the assertion on line 113 should fail
    TSV_LINE=$(grep -A 9 "${LDC_CORPUS}" ${DOWNLOAD_FILE})
    CORPUS_URLS=($(echo "${TSV_LINE}" | grep "/download/" | sed "s/.*download\/\(.*\)' title='Download Corpus'.*/\1/"))
    CORPUS_NAMES=($(echo "${TSV_LINE}" | grep "<br/>" | sed 's/\(.*\)<br\/>.*/\1/'))

    if [ "${#CORPUS_URLS[@]}" -ne "${#CORPUS_NAMES[@]}" ]; then
        log "Found differing #s of download URLs and corpora names for ${LDC_CORPUS}"
        log "${CORPUS_URLS[@]}"
        log "${CORPUS_NAMES[@]}"
        exit 1
    fi

    for i in "${!CORPUS_URLS[@]}"; do
        log "downloading /download/${CORPUS_URLS[i]} to ${CORPUS_NAMES[i]}.tgz"

        echo

        LDC_CORPUS_FILENAME="${LDC_CORPUS}__${CORPUS_URLS[i]}__${CORPUS_NAMES[i]}.tgz"

        DOWNLOAD_URL="/download/${CORPUS_URLS[i]}"

        curl -k --user "${EMAIL}:${PASSWORD}" "${BASE_URL}${DOWNLOAD_URL}" > "${LDC_CORPUS_FILENAME}"

        echo

        FILE_TYPE=$(file -i -b "${LDC_CORPUS_FILENAME}" | cut -d ';' -f 1)

        if [[ "${FILE_TYPE}" == "text/html" ]]; then
        	log "Encountered an error downloading ${LDC_CORPUS_FILENAME}. The downloaded file is HTML, but was expected to be .tgz"
        	exit 1
        fi
    done
done

echo
