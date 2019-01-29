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
	# For each category:
	# 4096 pics from train/ will go to places10/train
	#  128 pics from train/ will go to places10/val
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places10/ scripting/places10.txt train 4096 --bname val --bsize 128
	#   96 pics from val/   will go to places10/test
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places10/ scripting/places10.txt test 96
	@echo  '... done.'
	@echo 'Please run `make run` to train the network.'


dataset: setup download extract
	@echo "Downloading, extracting and splitting dataset."
	@echo "Run `make split` to split the dataset"

run:
	python3 loader.py config/places10.yaml


zip_out:
	zip --quiet --recurse-paths out.zip out/
	#rm -rf out/*
	@echo "Zipped all files in ./out into out.zip"

clean:
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm colorized-*.jpg


places16:
	@echo 'Places16 - splitting dataset ...'
	# For each category:
	# 4096 pics from train/ will go to places16/train
	#  128 pics from train/ will go to places16/val
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places16/ scripting/places16.txt train 4096 --bname val --bsize 128
	#   96 pics from val/   will go to places16/test
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places16/ scripting/places16.txt test 96
	@echo  '... done.'
	@echo 'Please run `python3 loaader.py config/places16.yaml` to train the network.'
    
    
places13:
	@echo 'Places13 - splitting dataset ...'
	# For each category:
	# 4096 pics from train/ will go to places13/train
	#  128 pics from train/ will go to places13/val
	python3 scripting/split-dataset.py data/places365_standard/train/ data/places13/ scripting/places13.txt train 4096 --bname val --bsize 128
	#   96 pics from val/   will go to places13/test
	python3 scripting/split-dataset.py data/places365_standard/val/ data/places13/ scripting/places13.txt test 96
	@echo  '... done.'
	@echo 'Please run `python3 loaader.py config/places13.yaml` to train the network.'
