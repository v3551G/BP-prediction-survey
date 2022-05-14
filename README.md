# Machine learning and deep learning for blood pressure prediction: A methodological review from multiple perspectives

## Introduction
This repository contains the code for reproducing the results of the upcoming paper titled "Machine learning and deep learning for blood pressure prediction: A methodological review from multiple perspectives.

The code was written using TensorFlow 2.4.0 and Python 3.8. All experiments were performed on an Ubuntu Server equipped with RTX 3080 TI GPU. If you have any questions, please contact: `masterqkk@outlook.com`.

## Project Overview
This repository contains code for reproducing the results in the upcoming paper mentioned above. Specifically, we mainly analyzed the results of the trained ResNet model on MIMIC III dataset based on different splitting strategies.

The Dir 'models' contains the definition of model architecture used. If you want to use other model architecture for trainnig BP predictive model, please place the model architecture definition file under this Dir.

The Dir 'analyze' contains scripts for analyzing the experimental results, including the averaged evaluation results, significant test, etc.  

The following table summarizes the function of several main python files.

|     | Script                            | Description                                                                                                      |
|-----|-----------------------------------|------------------------------------------------------------------------------------------------------------------|
| 1   |     `define_ResNet_1D.py`         | Definition of the ResNet model architecture for BP estimation |
| 2   | `1download_mimic_iii_records.py`  | Script for downloading data from the MIMIC-III database                                                          |
| 3   | `2prepare_MIMIC_dataset.py`       | Script for preprocessing (signal filtering, segmentation, abnormal exclusion) the data.                          |
| 4   | `3split_and_save_dataset.py`      | Script for dividing the processed dataset into trainnig, validation and test test, and save it as tfrecord file. |
| 5   | `4training_model_and_evaluation.py` | Script for training a ResNet model for BP estimation and model evaluation.                                       |

## Run the experiments using command
### Downloading data from the MIMIC-III database
The script `1download_mimic_iii_records.py` can be used to download the records used for training BP estimation mdoel. The specific record names are provided in the file `MIMIC-III_ppg_dataset_records.txt`. The script can be called from the command line using the command
```
python3 1download_mimic_iii_records.py [-h] --input --output

positional arguments:
  --input       File containing the names of the records downloaded from the MIMIC-III DB
  --output      Folder for storing downloaded MIMIC-III records

Demo:
  python3 1download_mimic_iii_records.py --input ./MIMIC-III_ppg_dataset_records.txt  --output mimiciii
           


```
Noe that this requires a long time and the downloaded data is very large.

### Preparing the PPG dataset
The script `2prepare_MIMIC_dataset.py` is used to preprocess (signal filtering, segmentation, abnormal exclusion) the downlaoded data.

```
usage: 2prepare_MIMIC_dataset.py [-h] [--win_len WIN_LEN] [--win_overlap WIN_OVERLAP] [--maxsampsubject MAXSAMPSUBJECT]
                                [--maxsamp MAXSAMP] [--save_bp_data SAVE_BP_DATA]
                                datapath output

positional arguments:
  datapath              Path containing data records downloaded from the MIMIC-III database
  output                Target .h5 file

optional arguments:
  -h, --help            show this help message and exit
  --win_len WIN_LEN     PPG window length in seconds
  --win_overlap WIN_OVERLAP
                        ammount of overlap between adjacend windows in fractions of the window length (0...1)
  --maxsampsubject MAXSAMPSUBJECT
                        Maximum number of samples per subject
  --maxsamp MAXSAMP     Maximum total number os samples in the dataset
  --save_ppg_data SAVE_PPG_DATA
                        0: save BP data only; 1: save PPG and BP data

Demo: 
  python3 2prepare_MIMIC_dataset.py ./mimiciii ./processed
    
Note that the processed dataset has been attached in the dir 'processed'.
```

### Splitting the data into training, validation and test set for model training, validation and evaluation
The script `3split_and_save_dataset.py` is used to divide the dataset based on different splitting strategies. 
The splitted sets will be stored separately for training, validation and testset in .tfrecord files under the dir 'splits'.

```
usage: 3split_and_save_dataset.py [-h] [--ntrain NTRAIN] [--nval NVAL] [--ntest NTEST] [--divbysubj DIVBYSUBJ] input output

positional arguments:
  input                 Path to the .h5 file containing the dataset
  output                Target folder for the .tfrecord files

optional arguments:
  -h, --help            Show this help message and exit
  --ntrain NTRAIN       Number of samples in the training set (default: 9e5)
  --nval NVAL           Number of samples in the validation set (default: 3e5)
  --ntest NTEST         Number of samples in the test set (default: 3e5)
  --split_strategy      's'/'si'/'sir' denote sample level splitting strategy, 'r' denotes record level splitting strategy. for the detail meaning of these flags, please refer the file: 3split_and_save_dataset.py.
  --enlarge_ratio       The number of records used is 750 by default, and the actual number of records used in 750 * enlarge_ratio.
  --random_seed         Corresponding to different splits, which is used to control the randomness in data spliting (default: 0)

Demo: 
python3 3split_and_save_dataset.py  --split_strategy 's'  ./processed/MIMIC-III_ppg_dataset.h5  ./splits

python3 3split_and_save_dataset.py  --enlarge_ratio 3.0 --split_strategy 'r' --random_seed 0 ./processed/MIMIC-III_ppg_dataset.h5  ./splits


```
### Training ResNet model for BP estimation
The script `4training_model_and_evaluation.py` trains a ResNet model for BP estimation. After the experiment is finished, the model checkpoints are stored under the dir 'ckpts', the result files are stored under the dir 'results'.

```
usage: 4training_model_and_evaluation.py [-h] [--arch ARCH] [--lr LR] [--batch_size BATCH_SIZE] [--winlen WINLEN] [--epochs EPOCHS]
                                 [--gpuid GPUID]
                                 ExpName datadir resultsdir chkptdir

positional arguments:
  ExpName               unique name for the training
  datadir               folder containing the train, val and test subfolders containing tfrecord files
  resultsdir            Directory in which results are stored
  chkptdir              directory used for storing model checkpoints

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           neural architecture used for training (alexnet (default), resnet, slapnicar, lstm)
  --lr LR               initial learning rate (default: 0.003)
  --batch_size BATCH_SIZE
                        batch size used for training (default: 32)
  --winlen WINLEN       length of the ppg windows in samples (default: 875)
  --epochs EPOCHS       maximum number of epochs for training (default: 60)
  --gpuid GPUID         GPU-ID used for training in a multi-GPU environment (default: None)
  --verbose             0/1: doesnot/does output log in stdandard I/O, 2: output log every epoch 

Demo: 
[for regular experiment]
python3 4training_model_and_evaluation.py --arch resnet --random_seed 0 --split_strategy 'r' exp_resnet ./splits ./results ./ckpts

[for experiment with enlarge_ratio !=1], 0.5 -> 64, 1.0 ->128, 2.0 ->256, 3.0 -> 384, 4.0 ->512, 
python3 4training_model_and_evaluation.py  --verbose 2 --arch resnet --random_seed 2 --split_strategy 'r' --enlarge_ratio 4.0 --batch_size 512 exp_resnet ./splits ./results ./ckpts

