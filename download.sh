#!/bin/bash/

mkdir -s data/

if [ ! -f places365standard_easyformat.tar ]; then
    echo "Downloading dataset ..."
    
    wget -nc http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

    echo "... done."
    echo "Extracting to data/ ..."

    tar -xvf places365standard_easyformat.tar -C data/

    echo '... done!'
fi


echo 'Splitting dataset ...'

python3 scripting/split-dataset.py categories10.txt

echo '... done!'
echo 'Please run run.sh to train the network.'

