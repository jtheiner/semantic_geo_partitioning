
# Classification

## Test on Already Trained Models

### Download Pretrained Models

We provide the EfficientNet-B4 checkpoints for the reproduced s2(M,f*) and our *SemP(100,125,250, f)* variants including the respective partitioning and class mapping to the training data:
```sh
export dl_path="data/wacv22_checkpoints_efficientnet"
mkdir -p $dl_path/semp_100_125_250
wget https://tib.eu/cloud/s/F5o6MP7y7AbLSpi/download/semp_100_125_250.zip -O $dl_path/semp_100_125_250.zip
unzip $dl_path/s2.zip -d $dl_path/semp_100_125_250/

mkdir -p $dl_path/s2
wget https://tib.eu/cloud/s/4EYyook2nm6EsNQ/download/s2.zip -O $dl_path/s2.zip
unzip $dl_path/s2.zip -d $dl_path/s2/
```

### Results

```
export baseckpt="data/wacv22_checkpoints_efficientnet/s2"
python geo_classification/evaluate_testsets.py --config $baseckpt/hparams.yaml --checkpoint $baseckpt/base.ckpt
export baseckpt="data/wacv22_checkpoints_efficientnet/semp_100_125_250"
python geo_classification/evaluate_testsets.py --config $baseckpt/hparams.yaml --checkpoint $baseckpt/base.ckpt
```

|         testset         | checkpoint   |   acc@1km |   acc@25km |   acc@200km |   acc@750km |   acc@2500km |
|-------------------------|--------------|-----------|------------|-------------|-------------|--------------|
| Im2GPS3k | SemP({100, 125, 250}, f)    |     12.5 |     31.4 |      42.7 |      57.3 |       72.0 |
| Im2GPS3k | s2(M, f*)                   |     11.7 |     31.5 |      41.9 |      56.1 |       70.6 |
| Im2GPS   | SemP({100, 125, 250}, f)    |     16.9 |     45.6 |      57.4 |      72.6 |       85.7 |
| Im2GPS   | s2(M, f*)                   |     15.6 |     41.8 |      54.9 |      71.3 |       81.4 |
| yfcc4k   | SemP({100, 125, 250}, f)    |     9.4  |     20.3 |      30.5 |      44.8 |       61.3 |
| yfcc4k   | s2(M, f*)                   |     7.5  |     19.7 |      28.5 |      42.5 |       59.1 |



### Official Testsets
Download and prepare testsets (Im2GPS, Im2GPS3k, YFCC4k):

```sh
mkdir -p ../data/images/testsets
cd ../data/images/testsets
mkdir -p im2gps/img
mkdir -p im2gps3k/img
mkdir -p YFCC4k/img

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

### Testset File structure

Expected folder structure: `data/images/testsets/<testsetname>/<img>`
Expected file format: `<testsetname>/*.csv`

The header of each CSV file contains at least [`img_id`, `latitude`, `longitude`], where `img_id` is also the image filename, located in the subdirectory `img` of the CSV folder.


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
