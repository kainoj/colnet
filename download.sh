#!/bin/bash

mkdir -p data/

if [ ! -f places365standard_easyformat.tar ]; then
    echo "Downloading dataset ..."
    echo
    
    wget -nc http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

    echo
    echo "... done."
    echo "Extracting to data/ ..."

    tar -xvf places365standard_easyformat.tar -C data/

    echo '... done!'
fi


echo 'Splitting dataset ...'
echo

python3 scripting/split-dataset.py scripting/categories10.txt

echo
echo '... done!'
echo 'Please run run.sh to train the network.'

