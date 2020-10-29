# Crack Detection as a Weakly-Supervised Problem: Towards Achieving Less Annotation-Intensive Crack Detectors

Official repository of our [ICPR2020 paper]().

You will find the following in this repository:

- A script that downloads Aigle, CFD, and DeepCrack datasets
- URL for the low quality annotations repo (Rough, Rougher, Dil1-4 annotations used in the paper)
- Codes to generate your own synthetic annotations
- Setup script for the two crack detector OSS's used in the paper (DeepCrack and DeepLab v3)



## Requirements

Experiments were conducted on Ubuntu 18.04 with Python 3.6.9 and CUDA9. Other dependencies are summarized in ```requirements.txt```.


## Data Preparation

### Downloading the original annotations

In our experiments, the following datasets are used for training and testing:

- Aigle
- Crack Forest Dataset (CFD)
- DeepCrack Dataset (DCD)

These datasets are publicly available through different websites and GitHub repos. Run the following line to download the images and annotations

```shell
./tools/download.sh
```

Downloaded datasets should be available under ```data/*_detailed```, where ```*``` stands for the name of the dataset.



### Low Quality Annotation Repo

The proposed method was tested with various low quality annotations. Both manual and synthetic annotations are available through the [Zenodo repo](). After downloading the datasets, please locate them under ```data``` directory. Also note that the downloaded dataset only contains the annotations. Please run the following line to copy the RGB input images from the ```data/*_detailed``` directories.

```shell
python tools/data_gen.py --fill
```

The Zenodo repo also contains ```pascal_voc_seg``` folder, which contains the pretrained Xception backbone for DeepLab. Place the folder under ```tools/model_supp/deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug```.



### Synthesizing your own dataset

You may want to synthesize your own low quality annotations. This can be done with the following line.

```shell
# generate synthetic dataset for the Aigle dataset, with dilation values 1, 2, 3, and 4
python tools/data_gen.py --synth_dil_anno --anno_type 1 2 3 4 --dataset_name aigle
```

```--anno_type``` specifies the dilation value (```n_{dil}``` in the paper). Larger value implies lower quality annotation.
```--dataset_name``` specifies the dataset name. Set ```all``` for synthesizing annotations for aigle, cfd, and deepcrack.



### Formatting and copying the datasets

:warning: **This step needs to be done after the crack detectors are downloaded- i.e. after running the ```tools/setup_models.sh``` script**

The datasets under ```data``` directory need to be formatted to be used by different crack detectors. This can be done with the following lines.

```shell
# format and dispatch detailed and dil1 annotations for used by DeepCrack
python tools/data_gen.py --deepcrack --anno_type detailed 1 --dataset_name all

# format and dispatch dil1 dil2 and rough annotations for used by DeepLab
python tools/data_gen.py --deeplab --anno_type 1 2 rough --dataset_name all
```

This script also copies the formatted annotations to the correct data directories within the downloaded repos. For DeepCrack repo, the data directory is ```${DEEPCRACK_REPO}/datasets```, and for DeepLab repo, it is ```${DEEPLAB_REPO}/research/deeplab/datasets/data``` and ```${DEEPLAB_REPO}/research/deeplab/datasets/data/tfrecords```.



## Setting up the Crack Detectors

Run the following script to download and modify the crack detector repos.

```shell
./tools/setup_models.sh
```

This script should correctly set up the two crack detector repos under ```models``` directory. Please refer to the following sections for more details on what the script does.

:warning: **Do not forget to copy the datasets to the crack detector repos before training them. You can copy the dataset by following the instructions outlined in "Formatting and copying the datasets" section above.**



### [DeepCrack](https://github.com/yhlleo/DeepSegmentor)

We use the ```50440b52ddaf49cf54c2415e6b40646a7601c219``` commit of the DeepCrack repo. After the repository is cloned and checked out, the following files are modified (i.e. copied from ```tools/model_supp/deepcrack``` directory):

```
models/deepcrack_model.py
options/train_options.py
```

You can see the details of the modifications in ```tools/model_supp/deepcrack```.

#### Training

You can train the DeepCrack model by running ```scripts/train_deepcrack.sh``` from ```models/deepcrack``` directory. Modify the script accordingly to train the model with various annotations.
The training results are saved under the ```checkpoints``` directory.

#### Evaluation

You can evaluate the DeepCrack model by running ```scripts/test_eval.sh``` from ```models/deepcrack``` directory. Modify the script accordingly to evaluate the model outputs for various annotations.
The training results are saved under the ```checkpoints``` directory.

For more details on training and evaluation, please refer to the original repository.



### [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)

We use the ```0a161121852ee5f34b939279d54b5d3e231ca501``` commit of the DeepLab repo (sorry it is an old commit, the recent repository uses TF v2 instead of v1). After the repository is cloned and checked out, the following files are modified (i.e. copied from ```tools/model_supp/deepcrack``` directory):

```
research/deeplab/datasets/segmentation_dataset.py
research/deeplab/utils/train_utils.py
research/deeplab/train.py
research/deeplab/export_model.py
research/deeplab/model.py
```

You can see the details of the modifications in ```tools/model_supp/deeplab```.

#### Training

You can train the DeepLab model by running ```scripts/train.sh``` from ```models/deeplab/research/deeplab``` directory. Modify the script accordingly to train the model with various annotations.
The training results are saved under the ```outputs``` directory.

#### Evaluation

You can evaluate the DeepLab model by running ```scripts/inference.sh``` from ```models/deeplab/research/deeplab``` directory. Modify the script accordingly to evaluate the model outputs for various annotations.
The training results are saved under the ```outputs``` directory.

For more details on training and evaluation, please refer to the original repository.



### Inoue et. al.

Unfortunately, we cannot release this code due to company confidentiality reasons.



### Evaluating with the Micro Branch

To evaluate the trained models with the Micro Branch, copy the evaluation results from the crack detector repos to the ```eval/results``` directory. Results are stored under ```models/deepcrack/checkpoints``` for DeepCrack, and results are stored under ```models/deeplab/research/deeplab/outputs``` for DeepLab.

Evaluation can be done as follows:

```shell
python eval/micro_eval.py
```

This script first applies the Micro Branch output to the crack detector outputs and stores the results under ```eval/results/${MODEL_RESULT_FOLDERNAME}/cv_output_none```.
Then the script evaluates the result and outputs ```eval_cv_output_none_dil0.txt``` and ```eval_test_output_dil0.txt```, which correspond to results with Micro Branch and without Micro Branch, respectively.

The result for all model results can be aggregated using the ```gen_table.py``` script. It outputs a csv file named  ```results.csv``` under the main directory.



## Citation

Please consider citing our paper if it helps your research:
​```
@inproceedings{inoue2020crack,
    title={Crack Detection as a Weakly-Supervised Problem: Towards Achieving Less Annotation-Intensive Crack Detectors},
    author={Inoue, Yuki and Nagayoshi, Hiroto},
    booktitle={International Conference on Pattern Recognition (ICPR)},
    year={2020},
}​```



## Acknowledgement

Our project is built from the following repositories. Thanks you for your great works!

- Models
  - [DeepCrack](https://github.com/yhlleo/DeepSegmentor)
  - [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)
- Datasets
  - [Aigle]()
  - [Crack Forest Dataset]()
  - [DeepCrack](https://github.com/yhlleo/DeepCrack)