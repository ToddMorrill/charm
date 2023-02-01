#!/bin/bash
# Wrapper script to call docker, which in turn calls Ruby script to process URL list.

odir=`pwd`/out
if [[ $# -ne 1 || ! -f $1 ]]
then
  echo "Usage: ./get_urls.sh <URL_list_file>"
  exit
fi
docker build --build-arg "urlfile=$1" -t ldc_dl_tool .
docker run -it -v $odir:/tmp --rm ldc_dl_tool $1
