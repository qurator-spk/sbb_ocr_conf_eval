#/usr/bin/env bash

prefix=scratch/ppn_list.batch

for ppn_list in $prefix*;do
  batch_id=${ppn_list//$prefix/}
  sbb_ocr ppn2confs --output confs.$batch_id.csv --format csv $(cat $ppn_list)
done
