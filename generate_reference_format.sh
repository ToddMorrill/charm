#!/bin/bash

E13="LDC2022E13_CCU_TA1_Mandarin_Chinese_Development_Metadata_and_Associated_Files_V1.0"
E18="LDC2022E18_CCU_TA1_Mandarin_Chinese_Development_Annotation_V1.0"

# make a scoring index file
mkdir -p $E18/index_files
awk -F '\t' '{ print $1 }' $E18/docs/segments.tab | sort -r | uniq > $E18/index_files/LDC2022E18-V1.scoring.index.tab

# make a file_info file
echo -e "file_uid\ttype\tlength" > $E18/docs/file_info.tab
for i in $(cat $E18/index_files/LDC2022E18-V1.scoring.index.tab | grep -v "file_id");
  do if grep -q -w $i $E13/docs/file_info.tab;
  then cat $E13/docs/file_info.tab| grep -w $i | awk -F '\t' 'BEGIN {OFS = FS} { print $5,$3,10000 }' >> $E18/docs/file_info.tab;
  else echo -e "$i\tvideo\t10000" >> $E18/docs/file_info.tab;
  fi;
done
