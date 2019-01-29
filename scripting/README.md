`split-datset.py` splits given dataset into two disjoint subsets. `.txt` file specifies which categories should be considered.

Example:
```bash
# For each category in `places10.txt`
# 4096 pics from train/ will go to places10/train
#  128 pics from train/ will go to places10/val
python3 scripting/split-dataset.py data/places365_standard/train/ data/places10/ scripting/places10.txt train 4096 --bname val --bsize 128
#   96 pics from val/   will go to places10/test
python3 scripting/split-dataset.py data/places365_standard/val/ data/places10/ scripting/places10.txt test 96
```



More:
```bash
python3 scripting/split-dataset.py --help
```


