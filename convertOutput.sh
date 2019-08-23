#!/bin/bash

## Convert output tiff files into jpg ones
## Move raw tiff files into /tmp
## Doron Adler, @Norod78
inputFolder=~/Documents/'Shared Playground Data'
pushd "$inputFolder"
for i in *.tiff; do sips -s format jpeg -s formatOptions 70 "${i}" --out "${i%tiff}jpg"; done
open .
mv *.tiff /tmp/
popd
