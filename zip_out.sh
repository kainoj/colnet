#!/bin/bash

zip -r out.zip out/
rm -rf out/
mkdir out/

echo "Zipped all files in ./out into out.zip"
