# Machine learning and deep learning for blood pressure prediction: A methodological review from multiple perspectives

## Introduction
This repository contains the code for reproducing the results of the upcoming paper titled "Machine learning and deep learning for blood pressure prediction: A methodological review from multiple perspectives, and other useful resources.

The code was written using TensorFlow 2.4.0 and Python 3.8. All experiments were performed on an Ubuntu Server equipped with RTX 3080 TI GPU. If you have any questions, please contact: `masterqkk@outlook.com`.

## Project Overview
This repository contains code for reproducing the results in the upcoming paper mentioned above and other useful resources. 

### Code

Specifically, we mainly analyzed the results of the trained ResNet model on MIMIC III dataset based on different splitting strategies.

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

### Other resources
we list several representative surveys, research papers and the open-source implementations.

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
    
Note that the processed dataset ca not be uploaded to Github due to the limits of maximum allowed size. The processed data can be acquired at reasionable request via the email: masterqkk@outlook.com.
```
![](/home/masterqkk/Desktop/xx.png)

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
```

## Other useful resources
### Representative surveys
                                       
|     | Paper                             | Publication                                   | URL |
|-----|-----------------------------------|-----------------------------------------------|-----|
| 1   | Evaluation of the accuracy of cuffless blood pressure measurement devices: Challenges and proposals         | Hypertension                                  |   https://doi.org/10.1161/HYPERTENSIONAHA.121.17747  |
| 2   | Accuracy of cuff-measured blood pressure: systematic reviews and meta-analyses  | Journal of the American College of Cardiology |  https://www.jacc.org/doi/abs/10.1016/j.jacc.2017.05.064   |
| 3   | Blood pressure measurement: clinic, home, ambulatory, and beyond     | American Journal of Kidney Diseases           |  https://doi.org/10.1053/j.ajkd.2012.01.026   |
| 4   | Cuffless single-site photoplethysmography for blood pressure monitoring    | Journal of Clinical Medicine                  |  https://pubmed.ncbi.nlm.nih.gov/32155976/   |
| 5   | The Machine Learnings Leading the Cuffless PPG Blood Pressure Sensors Into the Next Stage | IEEE Sensors Journal                          |   https://ieeexplore.ieee.org/abstract/document/9406011  |
| 6   | A review of machine learning techniques in photoplethysmography for the non-invasive cuff-less measurement of blood pressure | Biomedical Signal Processing and Control      |  https://doi.org/10.1016/j.bspc.2020.101870  |
| 7   | A review of machine learning in hypertension detection and blood pressure estimation based on clinical and physiological data | Biomedical Signal Processing and Control      |   https://doi.org/10.1016/j.bspc.2021.102813 |
| 8   | Oscillometric blood pressure estimation: Past, Present, and Future | IEEE Reviews in Biomedical Engineering        |   https://ieeexplore.ieee.org/abstract/document/7109154 |
| 9   | Cuffless blood pressure monitors: Principles, standards and approval for medical use | IEICE Transactions on Communications          |   https://doi.org/10.1587/transcom.2020HMI0002 |
| 10  | Smartphones and video cameras: Future methods for blood pressure measurement | Frontiers in Digital Health                   |   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8633391/ |
| 11  | A survey: From shallow to deep machine learning approaches for blood pressure estimation using biosensors | Expert Systems with Applications              |   https://doi.org/10.1016/j.eswa.2022.116788 |

### Open-source implementations
                                       
|     | Paper                                                              | URL                                                                                                                                      |
|-----|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | UNet-based model for reconstructing ABP waveform  using PPG signal | https://github.com/nibtehaz/PPG2ABP                                                                                                      |
| 2   | Seq2Seq model with attention mechanism for reconstructing ABP waveform using PPG signal | processed dataset: https://doi.org/10.5281/zenodo.4598938, code: https://github.com/AguirreNicolas/PPG2IABP                              |
| 3   | MLP-Mixer based model for predicting BP using PPG and ECG signals  | data\&code: https://github.com/ marshb/MLP-BP                                                                                            |
| 4   | Dendritic neural model for predicting BP using PPG and ECG signals   | code: http://www.dnm.net.cn/index.html                                                                                                   |
| 5   | Domain-adversarial model for predicting BP using bioimpedance sensors  | code: https://github.com/stmilab/cufflessbp\_dann                                                                                        |
| 6   | Spectro-temporal neural network model for predicting BP using PPG signal  | code: https://github.com/gasper321/bp-estimation-mimic3                                                                                  |
| 7   | Convolution-based model for predicting BP using PPG  and rPPG signals  | processed dataset: https://zenodo.org/record/5590603, code: https://github.com/Fabian-Sc85/non-invasive-bp-estimation-using-deep-learning |
| 8   | Deep RNN model for predicting long-term BP using PPG and ECG signals  | code: https://github.com/psu1/DeepRNN                                                                                                    |
| 9   | Two-stage hybrid model for predicting BP using PPG signal     | code: https://github.com/jesmaelpoor/Blood-pressure-estimation--Deep-multistage-model                                                    |
| 10  |  SVM model for predicting BP using PPG signal  | code: https://github.com/thmedialab/DataDrivenBP                                                                                         |
| 11  | CycleGAN based model with federated learning for predicting ABP waveform  using PPG signal    | https://github.com/Brophy-E/T2TGAN                                                                                                       |
| 12  | LSTM based model for predicting BP using PPG and ECG signals   | code: https://github.com/ploymel/estimateBP                                                                                              |
| 13  | V-Net based model for predicting ABP waveform using PPG and ECG signals    | https://github.com/brianhill11/ABPImputation                                                                                             |
| 14  | Random forest model with genetic algorithm for predicting BP using PPG and ECG signals  | data\&code: https://github.com/jeya-maria-jose/Cuff\_less_BP\_Prediction                                                                                                                             |

### Related papers on BP estimation 
                                      
                                      
                                       
|     | Paper                                                              | URL                                                                                                                                      |
|-----|--------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Cuffless differential blood pressure estimation using smart phones                                                                                                                   |     |
| 2 | Combined deep {CNN-LSTM} network-based multitasking learning architecture for noninvasive continuous blood pressure estimation using difference in {ECG-PPG} features                |     |
| 3 | Concatenated Convolutional Neural Network Model for Cuffless Blood Pressure Estimation Using Fuzzy Recurrence Properties of {PPG} Signals                                            |     |
| 4 | Analysis of pulse arrival time as an indicator of blood pressure in a large surgical biosignal database: recommendations for developing ubiquitous blood pressure monitoring methods |     |
| 5 | Cuff-less high-accuracy calibration-free blood pressure estimation using pulse transit time                                                                                          |     |
| 6 | Non-invasive blood pressure estimation using phonocardiogram                                                                                                                         |     |
| 7 | Nonlinear cuffless blood pressure estimation of healthy subjects using pulse transit time and arrival time                                                                           |     |
| 8 | Pervasive blood pressure monitoring using {P}hotoplethysmogram ({PPG}) sensor                                                                                                        |     |
| 9 | Wireless medical sensor network for blood pressure monitoring based on machine learning for real-time data classification                                                            |     |
| 10 | Noninvasive classification of blood pressure based on photoplethysmography signals using bidirectional long short-term memory and time-frequency analysis                            |     |
| 11 | {PPG2ABP}: Translating photoplethysmogram ({PPG}) signals to arterial blood pressure ({ABP}) waveforms using fully convolutional neural networks                                     |     |
| 12 | An estimation method of continuous non-invasive arterial blood pressure waveform using photoplethysmography: A {U-Net} architecture-based approach                                   |     |
| 13 | Genetic Deep Convolutional Autoencoder Applied for Generative Continuous Arterial Blood Pressure via Photoplethysmography                                                            |     |
| 14 | Prediction of arterial blood pressure waveforms from photoplethysmogram signals via fully convolutional neural networks                                                              |     |
| 15 | Blood pressure morphology assessment from photoplethysmogram and demographic information using deep learning with attention mechanism                                                |     |
| 16 | Continuous Blood Pressure Estimation Using Exclusively Photopletysmography by {LSTM}-Based Signal-to-Signal Translation                                                              |     |
| 17 | Aortic blood pressure estimation: A hybrid machine-learning and cross-relation approach                                                                                              |     |
| 18 | Offline and online learning techniques for personalized blood pressure prediction and health behavior recommendations                                                                |     |
| 19 | Dempster--{S}hafer Fusion Based on a Deep Boltzmann Machine for Blood Pressure Estimation                                                                                            |     |
| 20 | Continuous blood pressure measurement from one-channel electrocardiogram signal using deep-learning techniques                                                                       |     |
| 21 | Towards accurate estimation of cuffless and continuous blood pressure using multi-order derivative and multivariate photoplethysmogram features                                      |     |
| 22 | Cuffless blood pressure estimation from {PPG} signals and its derivatives using deep learning models                                                                                 |     |
| 23 | Blood pressure estimation from {PPG} signals using convolutional neural networks and {S}iamese network                                                                               |     |
| 24 | Continuous blood pressure prediction using pulse features and {E}lman neural networks                                                                                                |     |
| 25 | Estimation and tracking of blood pressure using routinely acquired photoplethysmographic signals and deep neural networks                                                            |     |
| 26 | {PP-Net}: A deep learning framework for {PPG}-based blood pressure and heart rate estimation                                                                                         |     |
| 27 | A Personalised Blood Pressure Prediction System using {G}aussian Mixture Regression and Online Recurrent Extreme Learning Machine                                                    |     |
| 28 | Non-invasive estimate of blood glucose and blood pressure from a photoplethysmograph by means of machine learning techniques                                                         |     |
| 29 | Investigating the physiological mechanisms of the photoplethysmogram features for blood pressure estimation                                                                          |     |
| 30 | Deep learning models for cuffless blood pressure monitoring from {PPG} signals using attention mechanism                                                                             |     |
| 31 | End-to-end deep learning architecture for continuous blood pressure estimation using attention mechanism                                                                             |     |
| 32 | A non-invasive continuous cuffless blood pressure estimation using dynamic recurrent neural networks                                                                                 |     |
| 33 | Investigation on the effect of {Womersley} number, {ECG} and {PPG} features for cuff less blood pressure estimation using machine learning                                           |     |
| 34 | Estimation of the Blood Pressure Waveform using {E}lectrocardiography                                                                                                                |     |
| 35 | Neural Recurrent Approches to Noninvasive Blood Pressure Estimation                                                                                                                  |     |
| 36 | {K-SVD}: An algorithm for designing overcomplete dictionaries for sparse representation                                                                                              |     |
| 37 | Blood pressure prediction via recurrent models with contextual layer                                                                                                                 |     |
| 38 | Double Channel Neural Non Invasive Blood Pressure Prediction                                                                                                                         |     |
| 39 | A novel dynamical approach in continuous cuffless blood pressure estimation based on {ECG} and {PPG} signals                                                                         |     |
| 40 | Cuffless blood pressure estimation from electrocardiogram and photoplethysmogram using waveform based {ANN-LSTM} network                                                             |     |
| 41 | Long-term blood pressure prediction with deep recurrent neural networks                                                                                                              |     |
| 42 | Enhancement of blood pressure estimation method via machine learning                                                                                                                 |     |
| 43 | A new estimate technology of non-invasive continuous blood pressure measurement based on electrocardiograph                                                                          |     |
| 44 | Prediction of blood pressure after induction of anesthesia using deep learning: a feasibility study                                                                                  |     |
| 45 | Sparse characterization of {PPG} based on {K-SVD} for beat-to-beat blood pressure prediction                                                                                         |     |
| 46 | Sparse representation of photoplethysmogram using {K-SVD} for cuffless estimation of arterial blood pressure                                                                         |     |
| 47 | Predicting blood pressure from physiological index data using the {SVR} algorithm                                                                                                    |     |
| 48 | Features extraction for cuffless blood pressure estimation by autoencoder from photoplethysmography                                                                                  |     |
| 49 | A new approach based on dynamical model of the {ECG} signal to blood pressure estimation                                                                                             |     |
| 50 | Cuffless and continuous blood pressure estimation from the heart sound signals                                                                                                       |     |
| 51 | {PPG}-based blood pressure estimation using residual neural networks and spectrograms                                                                                                |     |
| 52 | Chest wearable apparatus for cuffless continuous blood pressure measurements based on {PPG} and {PCG} signals                                                                        |     |
| 53 | Beat-to-beat ambulatory blood pressure estimation based on random forest                                                                                                             |     |
| 54 | Estimation and Validation of Arterial Blood Pressure Using Photoplethysmogram Morphology Features in Conjunction With Pulse Arrival Time in Large Open Databases                     |     |
| 55 | Wearable piezoelectric-based system for continuous beat-to-beat blood pressure measurement                                                                                           |     |
| 56 | Beats-to-Beats Estimation of Blood Pressure During Supine Cycling Exercise Using a Probabilistic Nonparametric Metho                                                                 |     |
| 57 | Multi-sensor fusion approach for cuff-less blood pressure measurement                                                                                                                |     |
| 58 | Central Blood Pressure Estimation from Distal {PPG} Measurement using semiclassical signal analysis features                                                                         |     |
| 59 | Wearable cuff-less blood pressure estimation at home via pulse transit time                                                                                                          |     |
| 60 | Key Feature Selection and Model Analysis for Blood Pressure Estimation From Electrocardiogram, Ballistocardiogram and Photoplethysmogram                                             |     |
| 61 | A Data-Driven Model with Feedback Calibration Embedded Blood Pressure Estimator Using Reflective Photoplethysmography                                                                |     |
| 62 | A novel continuous blood pressure estimation approach based on data mining techniques                                                                                                |     |
| 63 | A Shallow {U-Net} Architecture for Reliably Predicting Blood Pressure ({BP}) from Photoplethysmogram ({PPG}) and Electrocardiogram ({ECG}) Signals                                   |     |
| 64 | Study of cuffless blood pressure estimation method based on multiple physiological parameters                                                                                        |     |
| 65 | Data-driven estimation of blood pressure using photoplethysmographic signals                                                                                                         |     |
| 66 | Machine Learning Method for Continuous Noninvasive Blood Pressure Detection Based on Random Forest                                                                                   |     |
| 67 | Pulse transit time-pulse wave analysis fusion based on wearable wrist ballistocardiogram for cuff-less blood pressure trend tracking                                                 |     |
| 68 | {PCA}-based multi-wavelength photoplethysmography algorithm for cuffless blood pressure measurement on elderly subjects                                                              |     |
| 69 | Personalized Blood Pressure Estimation Using Photoplethysmography: A Transfer Learning Approach                                                                                      |     |
| 70 | Accurate Fiducial Point Detection Using {H}aar Wavelet for Beat-by-Beat Blood Pressure Estimation                                                                                    |     |
| 71 | Blood pressure estimation using photoplethysmogram signal and its morphological features                                                                                             |     |
| 72 | Cuffless blood pressure monitoring from an array of wrist bio-impedance sensors using subject-specific regression models: Proof of concept                                           |     |
| 73 | Cuffless blood pressure estimation algorithms for continuous health-care monitoring                                                                                                  |     |
| 74 | Cuffless blood pressure estimation methods: physiological model parameters versus machine-learned features                                                                           |     |
| 75 | Conventional pulse transit times as markers of blood pressure changes in humans                                                                                                      |     |
| 76 | Photoplethysmography Fast Upstroke Time Intervals Can Be Useful Features for Cuff-Less Measurement of Blood Pressure Changes in Humans                                               |     |
| 77 | Enabling Wearable Pulse Transit Time-Based Blood Pressure Estimation for Medically Underserved Areas and Health Equity: Comprehensive Evaluation Study                               |     |
| 78 | Continuous {PPG}-based blood pressure monitoring using multi-linear regression                                                                                                       |     |
| 79 | Multi-level information fusion for learning a blood pressure predictive model using sensor data                                                                                      |     |
| 80 | Cuffless blood pressure estimation based on photoplethysmography signal and its second derivative                                                                                    |     |
| 81 | A novel neural network model for blood pressure estimation using photoplethesmography without electrocardiogram                                                                      |     |
| 82 | Learning-Based Model for Central Blood Pressure Estimation using Feature Extracted from {ECG} and {PPG} signals                                                                      |     |
| 83 | Featureless Blood Pressure Estimation Based on Photoplethysmography Signal Using {CNN} and {BiLSTM} for {IoT} Devices                                                                |     |
| 84 | Blood pressure estimation using photoplethysmography only: comparison between different machine learning approaches                                                                  |     |
| 85 | Non-invasive blood pressure estimation from {ECG} using machine learning techniques                                                                                                  |     |
| 86 | Predicting increased blood pressure using machine learning                                                                                                                           |     |
| 87 | Developing personalized models of blood pressure estimation from wearable sensors data using minimally-trained domain adversarial neural networks                                    |     |
| 88 | Continuous blood pressure estimation through optimized echo state networks                                                                                                           |     |
| 89 | Continuous blood pressure estimation from {PPG} signal                                                                                                                               |     |
| 90 | Cuffless blood pressure estimation based on haemodynamic principles: progress towards mobile healthcare                                                                              |     |
| 91 | Estimating blood pressure trends and the nocturnal dip from photoplethysmography                                                                                                     |     |
| 92 | {ECG}-Based Blood Pressure Estimation Using {M}echano-{E}lectric Coupling Concept                                                                                                    |     |
| 93 | Blood pressure estimation from appropriate and inappropriate {PPG} signals using A whole-based method                                                                                |     |
| 94 | Noninvasive cuffless blood pressure estimation using pulse transit time, {W}omersley number, and photoplethysmogram intensity ratio                                                  |     |
| 95 | {iPhone} {A}pp compared with standard blood pressure measurement--The {iPARR} trial                                                                                                  |     |
| 96 | Pulse transit time estimation of aortic pulse wave velocity and blood pressure using machine learning and simulated training data                                                    |     |
| 97 | Cuffless Blood Pressure Measurement Using Linear and Nonlinear Optimized Feature Selection                                                                                           |     |
| 98 | Photoplethysmogram intensity ratio: A potential indicator for improving the accuracy of {PTT}-based cuffless blood pressure estimation                                               |     |
| 99 | {PPG}-based systolic blood pressure estimation method using {PLS} and level-crossing feature                                                                                         |     |
| 100 | Cuffless blood pressure estimation using only a smartphone                                                                                                                           |     |
| 101 | Highly wearable cuff-less blood pressure and heart rate monitoring with single-arm electrocardiogram and photoplethysmogram signals                                                  |     |
| 102 | Intelligent Bio-Impedance System for Personalized Continuous Blood Pressure Measurement                                                                                              |     |
| 103 | Health data driven on continuous blood pressure prediction based on gradient boosting decision tree algorithm                                                                        |     |
| 104 | Cuff-less blood pressure measurement using supplementary {ECG} and {PPG} features extracted through wavelet transformation                                                           |     |
| 105 | Feasibility study for the non-invasive blood pressure estimation based on {PPG} morphology: Normotensive subject study                                                               |     |
| 106 | An Unobtrusive and Calibration-free Blood pressure estimation Method using photoplethysmography and Biometrics                                                                       |     |
| 107 | {InstaBP}: Cuff-less blood pressure monitoring on smartphone using single {PPG} sensor                                                                                               |     |
| 108 | Blood pressure estimation using on-body continuous wave radar and photoplethysmogram in various posture and exercise conditions                                                      |     |
| 109 | Discussion of Cuffless Blood Pressure Prediction Using Plethysmograph Based on a Longitudinal Experiment: Is the Individual Model Necessary?                                         |     |
| 110 | A highly sensitive pressure-sensing array for blood pressure estimation assisted by machine-learning techniques                                                                      |     |
| 111 | A new wearable device for blood pressure estimation using photoplethysmogram                                                                                                         |     |
| 112 | A non-invasive continuous blood pressure estimation approach based on machine learning                                                                                               |     |
| 113 | Estimating blood pressure from the photoplethysmogram signal and demographic features using machine learning techniques                                                              |     |
| 114 | Real-time cuffless continuous blood pressure estimation using deep learning model                                                                                                    |     |
| 115 | Assessment of deep learning based blood pressure prediction from {PPG} and {rPPG} signals                                                                                            |     |
| 116 | Assessment of Non-Invasive Blood Pressure Prediction from {PPG} and {rPPG} Signals Using Deep Learning                                                                               |     |
| 117 | Cuffless blood pressure estimation using single channel photoplethysmography: A two-step method                                                                                      |     |
| 118 | Noninvasive Cuffless Blood Pressure Estimation With Dendritic Neural Regression                                                                                                      |     |
| 119 | Improved {PPG}-based estimation of the blood pressure using latent space features                                                                                                    |     |
| 120 | Blood pressure estimation from photoplethysmogram using latent parameters}                                                                                                           |     |
| 121 | Non-invasive continuous blood pressure measurement based on mean impact value method, {BP} neural network, and genetic algorithm                                                     |     |
| 122 | Cuffless continuous blood pressure estimation from pulse morphology of photoplethysmograms                                                                                           |     |
| 123 | A deep learning approach to predict blood pressure from {PPG} signals                                                                                                                |     |
| 124 | An empirical study on predicting blood pressure using classification and regression trees                                                                                            |     |
| 125 | Continuous Blood Pressure Estimation From Electrocardiogram and Photoplethysmogram During Arrhythmias                                                                                |     |
| 126 | Calibration-Free Cuffless Blood Pressure Estimation Based on a Population With a Diverse Range of Age and Blood Pressure                                                             |     |
| 127 | Intermittent blood pressure prediction via multiscale entropy and ensemble artificial neural networks                                                                                |     |
| 128 | {SVR} ensemble-based continuous blood pressure prediction using multi-channel photoplethysmogram                                                                                     |     |
| 129 | An integrated blood pressure measurement system for suppression of motion artifacts                                                                                                  |     |
| 130 | Cuffless blood pressure estimation based on data-oriented continuous health monitoring system                                                                                        |     |
| 131 | Towards a portable-noninvasive blood pressure monitoring system utilizing the photoplethysmogram signal                                                                              |     |
| 132 | Feature exploration for knowledge-guided and data-driven approach based cuffless blood pressure measurement                                                                          |     |
| 133 | A noninvasive time-frequency-based approach to estimate cuffless arterial blood pressure                                                                                             |     |
| 134 | Smart phone based blood pressure indicator                                                                                                                                           |     |
| 135 | Cuffless blood pressure measurement using a smartphone-case based {ECG} monitor with photoplethysmography in hypertensive patients                                                                                                                                                                                    |     |
| 136 | A Continuous Cuffless Blood Pressure Estimation Using Tree-Based Pipeline Optimization Tool                                                                                                                                                                                     |     |
| 137 | Estimation of Continuous Blood Pressure from {PPG} via a Federated Learning Approach                                                                                                                                                                                     |     |
| 138  | Novel Blood Pressure Waveform Reconstruction from Photoplethysmography using Cycle Generative Adversarial Networks      |     |
| 139  | Toward ubiquitous blood pressure monitoring via pulse transit time: theory and practice     |     |
| 140  | A review of methods for non-invasive and continuous blood pressure monitoring: Pulse transit time method is promising?     |     |
| 141  | Noninvasive and nonocclusive blood pressure estimation via a chest sensor     |     |
| 142  | PPG sensor contact pressure should be taken into account for cuff-less blood pressure measurement     |     |
| 143  |  A revised point-to-point calibration approach with adaptive errors correction to weaken initial sensitivity of cuff-less blood pressure estimation    |     |
| 144  | A linear regression model with dynamic pulse transit time features for noninvasive blood pressure prediction     |     |
| 145  | Accelerometric Method for Cuffless Continuous Blood Pressure Measurement     |     |
| 146  | Pulse transit time based continuous cuffless blood pressure estimation: A new extension and a comprehensive evaluation     |     |
| 147  | Cuff-less continuous measurement of blood pressure using wrist and fingertip photo-plethysmograms: Evaluation and feature analysis     |     |
| 148  | A novel approach to estimate blood pressure of blood loss continuously based on stacked auto-encoder neural networks     |     |
| 149  | Cuff-less continuous blood pressure measurement based on multiple types of information fusion     |     |
| 150  |  Optical blood pressure estimation with photoplethysmography and {FFT}-based neural networks    |     |
| 151  | Continuous blood pressure estimation based on two-domain fusion model     |     |
| 152  | A hybrid model for blood pressure prediction from a {PPG} signal based on {MIV} and {GA-BP} neural network     |     |
| 153  | A novel frequency domain method for estimating blood pressure from photoplethysmogram     |     |
| 154  | {MLP-BP}: A novel framework for cuffless blood pressure measurement with {PPG} and {ECG} signals based on {MLP-Mixer} neural networks      |     |
| 155  | Generalized deep neural network model for cuffless blood pressure estimation with photoplethysmogram signal only     |     |
| 156  | Energy-efficient Blood Pressure Monitoring based on Single-site Photoplethysmogram on Wearable Devices     |     |
| 157  | A clinical set-up for noninvasive blood pressure monitoring using two photoplethysmograms and based on convolutional neural networks     |     |
| 158  | Cuffless blood pressure estimation based on composite neural network and graphics information     |     |
| 159  | Blood Pressure Prediction by a Smartphone Sensor using Fully Convolutional Networks     |     |
| 159  | End-to-end blood pressure prediction via fully convolutional networks     |     |
| 160  | Cuffless and continuous blood pressure estimation from ppg signals using recurrent neural networks     |     |
| 161  | Continuous systolic and diastolic blood pressure estimation utilizing long short-term memory network     |     |
| 162  | Blood pressure prediction with multi-cue based {RBF} and {LSTM} model     |     |
| 163  | Prediction of blood pressure variability using deep neural networks     |     |
| 164  | A Novel Machine Learning-Based Systolic Blood Pressure Predicting Model     |     |
| 165  | Beat-to-beat continuous blood pressure estimation using bidirectional long short-term memory network     |     |
| 166  | Cuffless deep learning-based blood pressure estimation for smart wristwatches     |     |
| 167  | Novel Data Augmentation Employing Multivariate {G}aussian Distribution for Neural Network-Based Blood Pressure Estimation     |     |
| 168  | A multi-type features fusion neural network for blood pressure prediction based on photoplethysmography     |     |
| 169  | A hybrid neural network for continuous and non-invasive estimation of blood pressure from raw electrocardiogram and photoplethysmogram waveforms     |     |
| 170  | A multistage deep neural network model for blood pressure estimation using photoplethysmogram signals     |     |
| 171  | Non-invasive cuff-less blood pressure estimation using a hybrid deep learning model     |     |
| 172  | Photoplethysmography-Based Blood Pressure Estimation Using Deep Learning    |     |
| 173  |  {HYPE}: Predicting Blood Pressure from Photoplethysmograms in a Hypertensive Population    |     |
| 174  | Personalized effect of health behavior on blood pressure: Machine learning based prediction and recommendation     |     |
| 175  | Using Wearables and Machine Learning to Enable Personalized Lifestyle Recommendations to Improve Blood Pressure     |     |
| 176  | Homecare-oriented intelligent long-term monitoring of blood pressure using electrocardiogram signals     |     |
| 177  | Smartphones and Video Cameras: Future Methods for Blood Pressure Measurement     |     |
| 178  | Non-contact method of blood pressure estimation using only facial video     |     |
| 179  | Remote estimation of pulse wave features related to arterial stiffness and blood pressure using a camera      |     |
| 180  | Techniques for estimating blood pressure variation using video images    |     |
| 181  | Blood pressure estimation using video plethysmography    |     |
| 182  | Using imaging Photoplethysmography ({iPPG}) Signal for Blood Pressure Estimation     |     |
| 183  | Smartphone-based blood pressure measurement using transdermal optical imaging technology    |     |
| 184  | Remote {PPG} based vital sign measurement using adaptive facial regions    |     |
| 185  | Robust blood pressure estimation using an {RGB} camera    |     |
| 186  | The noninvasive blood pressure measurement based on facial images processing    |     |
| 187  | Introducing contactless blood pressure assessment using a high speed video camera    |     |
| 188  | Multi-point near-field {RF} sensing of blood pressures and heartbeat dynamics    |     |
| 189  | Blood Pressure States Transition Inference Based on Multi-State {M}arkov Model    |     |
| 190  |  Non-contact heart rate and blood pressure estimations from video analysis and machine learning modelling applied to food sensory responses: A case study for chocolate   |     |
| 191  | A blood pressure prediction method based on imaging photoplethysmography in combination with machine learning    |     |
| 192  | Deep generative model with domain adversarial training for predicting arterial blood pressure waveform from photoplethysmogram signal     |     |
| 193  | A Continuous Blood Pressure Estimation Method Using Photoplethysmography by {GRNN}-Based Model    |     |
| 194  | Estimation of arterial blood pressure waveform from photoplethysmogram signal using linear transfer function approach    |     |
| 195  |  Imputation of the continuous arterial line blood pressure waveform from non-invasive measurements using deep learning   |     |
| 196  |  Oscillometric blood pressure estimation based on deep learning   |     |
| 197  |   Deep belief networks ensemble for blood pressure estimation  |     |
| 198  | Combining bootstrap aggregation with support vector regression for small blood pressure measurement    |     |
| 199  | Statistical approaches based on deep learning regression for verification of normality of blood pressure estimates    |     |
| 200  |  Uncertainty in blood pressure measurement estimated using ensemble-based recursive methodology   |     |
| 201  | Ensemble methodology for confidence interval in oscillometric blood pressure measurements    |     |
| 202  | {GMM-HMM}-based blood pressure estimation using time-domain features    |     |
| 203  | Feature-based neural network approach for oscillometric blood pressure estimation    |     |
| 204  | Electrocardiogram-assisted blood pressure estimation    |     |
| 205  | Blood pressure estimation from time-domain features of oscillometric waveforms using long short-term memory recurrent neural networks    |     |
| 206  | Blood pressure estimation from beat-by-beat time-domain features of oscillometric waveforms using deep-neural-network classification models    |     |
| 207  | Deep {B}oltzmann regression with mimic features for oscillometric blood pressure estimation    |     |
| 208  | Blood pressure estimation using time domain features of auscultatory waveforms and {GMM-HMM} classification approach    |     |
| 209  | A Novel Automated Blood Pressure Estimation Algorithm Using Sequences of {K}orotkoff Sounds    |     |
| 210  | A novel deep learning based automatic auscultatory method to measure blood pressure    |     |
| 211  | Blood pressure estimation from photoplethysmogram using a spectro-temporal deep neural network    |     |
| 212  | An Adaptive Weight Learning-Based Multitask Deep Network for Continuous Blood Pressure Estimation Using {E}lectrocardiogram Signals    |     |
| 213  | Attention Mechanism-Based Convolutional Long Short-Term Memory Neural Networks to Electrocardiogram-Based Blood Pressure Estimation    |     |
| 214  | Photoplethysmogram-based blood pressure evaluation using Kalman filtering and neural networks    |     |



This repository is continuously updating ......
