setup:
	mkdir -p data out model

download:
	@echo "Downloading dataset ..."
	wget -nc http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
	@echo "... done."
    	
extract:
	@echo "Extracting to data/ (if doesn't exist) ..."
	tar --keep-old-files -xf places365standard_easyformat.tar -C data/
	@echo "... done."

split:
	@echo 'Splitting dataset ...'
	python3 scripting/split-dataset.py scripting/categories10.txt
	@echo  '... done.'
	@echo 'Please run `make run` to train the network.'


dataset: setup download extract split
	@"Downloading, extracting and splitting dataset."

run:
	python3 loader.py config/places10-full.yaml


zip_out:
	zip --quiet --recurse-paths out.zip out/
	rm -rf out/*
	@echo "Zipped all files in ./out into out.zip"
	
