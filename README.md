# Interpretable Semantic Photo Geolocation

<div align="center">  

[![Conference](https://img.shields.io/badge/WACV-2022-6b8bc7.svg?style=for-the-badge)](https://openaccess.thecvf.com/content/WACV2022/html/Theiner_Interpretable_Semantic_Photo_Geolocation_WACV_2022_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2104.14995-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2104.14995)
</div>


This repository contains a re-implementation of our paper [Interpretable Semantic Photo Geolocation](#citation).


## Semantic Partitioning (SemP)

Subpackage `semantic_partitioning` contains:
- [x] script for reverse geocoding
- [x] raw dataset visualization
- [ ] scripts to construct the *semantic partitioning* (SemP)

See [semantic_partitioning/README.md](semantic_partitioning/README.md) for details.

## Classification
Subpackage `geo_classification` contains:
- [x] script to train from scratch
- [x] evaluation pipeline including testsets
- [x] pretrained models (EfficientNet-B4):

See [geo_classification/README.md](geo_classification/README.md) for details.
    
## Extended MP-16 Dataset (EMP-16)
To overcome the need for a full installation of a reverse geocoder such as [Nominatim](https://nominatim.openstreetmap.org/), we provide the postprocessed output of the [reverse geocoding](semantic_partitioning/reverse_geocoding.py) for the MP-16 dataset[^1] along with the validation set (YFCC-Val26k) which originally comprising photos and respective GPS coordinates.
Both datasets are subsets of the YFCC100M dataset[^2] which are crawled from Flickr.

Further details: [semantic_partitioning/README.md](semantic_partitioning/README.md)

## Concept Influence

We provide the underlying functionality to compute the presented concept influence metric based on given semantic maps and attribution/explanation maps.
Please note, that the computation of both semantic maps and explanation maps are not part of this repository.

***
## Requirements
```sh
conda env create -f environment.yml
conda activate github_semantic_geo_partitioning
# cd in respective subpackages
cd semantic_partitioning
cd geo_classification
```

## Citation

```BibTeX
@InProceedings{Theiner_2022_WACV,
    author    = {Theiner, Jonas and M\"uller-Budack, Eric and Ewerth, Ralph},
    title     = {Interpretable Semantic Photo Geolocation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {750-760}
}
```

## Licence
This work is published under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007. For details please check the
LICENSE file in the repository.

## References
[^1]: Larson, M., Soleymani, M., Gravier, G., Ionescu, B., & Jones, G. J. (2017). The benchmarking initiative for multimedia evaluation: MediaEval 2016. IEEE MultiMedia, 24(1), 93-96.
[^2]: Thomee, B., Shamma, D. A., Friedland, G., Elizalde, B., Ni, K., Poland, D., ... & Li, L. J. (2016). YFCC100M: The new data in multimedia research. Communications of the ACM, 59(2), 64-73.