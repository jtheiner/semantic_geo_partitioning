
# Classification

## Test on Already Trained Models

### File structure

Expected folder structure: `data/images/testsets/<testsetname>/<img>`
Expected file format: `<testsetname>/*.csv`

The header of each CSV file contains at least [`img_id`, `latitude`, `longitude`], where `img_id` is also the image filename, located in the subdirectory `img` of the CSV folder.


### Official Testsets
Download and prepare testsets (Im2GPS, Im2GPS3k, YFCC4k):

```sh
mkdir -p ../data/images/testsets
cd ../data/images/testsets
mkdir -p im2gps/img
mkdir -p im2gps3k/img
mkdir -p YFCC4k/img

# metadata
wget TODO -O im2gps/im2gps_places365.csv
wget TODO -O im2gps3k/im2gps3k_places365.csv
wget TODO -O YFCC4k/yfcc4k-meta.csv

# download and move raw images into their respective folders
wget http://graphics.cs.cmu.edu/projects/im2gps/gps_query_imgs.zip -O im2gps/img2gps.zip
unzip im2gps/im2gps.zip -d .im2gps/img

```
Im2GPS3k and YFCC4k according to the instructons in https://github.com/lugiavn/revisiting-im2gps/

### Evaluation

By default all testsets are evaluated according to the respective `hparams.yaml` of a model checkpoint using:
```sh
python evaluate_testsets.py \
--config ../data/trained_models/<some_configuration>/hparams.yaml \
--checkpoint ../data/trained_models/<some_configuration>/<some_checkpoint>.ckpt
```

## Dataloader - Expected Output

In general, the training dataloader supplies a batch of images and a list of tensors containing class indices for multi-head classification. Thus, for training each batch has the format, `(torch.Size(B, 3, H, W), list([torch.Size(B), torch.Size(B), ...]))` while for validation *latitude* and *longitude* is concatinated `torch.Size(B), torch.Size(B))` to this list to log the great circle distance error also during training.

We use same format as in [https://github.com/TIBHannover/GeoEstimation#Training-from-Scratch](https://github.com/TIBHannover/GeoEstimation#Training-from-Scratch) to store and load the image data and subsequently join additionally required information as class indices or coordinates in [MsgPackIterableDatasetMultiTargetWithDynLabels](datasets/msgpack_dataset.py). 
Please note, that you can write your own dataset class if you don't want to be constrained to our implementation.


## Training from Scratch

See example config files for model and training details.
```sh
# supports pytorch lightning trainer flags
python train.py --config configs/example_s2.yml # baseline: s2(M, f*)
python train.py --config configs/example_semp.yml # SemP({100, 125, 250})
```

### Results on Provided Models:
TODO
