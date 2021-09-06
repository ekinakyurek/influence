# Few-shot LAMA

This code is based on LAMA(https://github.com/facebookresearch/LAMA).

## Preparation

Please create a virtual environment with python 3.7, and do 
```
pip install -r requirements.txt
```

Then, go to the LAMA/ directory, and
```
source ../add_path.sh
```
This changes the PYTHONPATH variable, please change it to match your directory

Now let's prepare the TREx data-set. Goto LAMA/data, and 
```
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
mv data/* .
rm -r data && rm data.zip
```
Note that the trex_multihop dataset is already in data/

A note on vocab: Instead of using a common vocab (as in LAMA), we directly use the vocab from the Roberta model.

