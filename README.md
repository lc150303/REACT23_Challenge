# Unified Transformer for Facial Reaction Generation
Official PyTorch code of [USTC-AC](https://ustc-ac.github.io/) for [REACT2023 Challenge](https://sites.google.com/cam.ac.uk/react2023/home).

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data preparation](#data-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)
7. [Pretrained model](#pretrained-models)

## Introduction
This repository is used by team [USTC-AC](https://ustc-ac.github.io/) for [REACT2023 Challenge](https://sites.google.com/cam.ac.uk/react2023/home).     
We propose a **Unified Transformer for Facial Reaction (UniFR)** framework, which can generate online and offline facial reactions by the same model.     
We have pack the conda environment, please follow [Installation](#installation) before running train or eval codes.

The main data stream of the proposed frameworks is as follows:
1. Preprocess the uncropped .mp4 file into facial features (AU, VA, and expression features), pose features, and audio features (MFCC and [GeMAPS]() features). Features are saved to `root/data/*.h5` where `*` is `train_shuffled`, `val_sequential`, or `test_sequential`. The appropriation matrix of train set will be saved to `root/data/train_appro_shuffled.npy`. **Reaction labels (facial attributes) are tokenized**. 

2. During training, the dataloader reads the `root/data/train_shuffled.h5` sequentially and provides real/fake listener reactions according to `root/data/train_appro_shuffled.npy`. The returned dict for each sample is:
```python
{
    'sample_name': sample_name,     # string
    's_exp': s_exp,     # (T, 512)
    's_AU': s_AU,       # (T, 25088)
    's_VA': s_VA,       # (T, 1408)
    's_pose': s_pose,   # (T, 1408)
    's_MFCC': s_MFCC,   # (T, 26)
    's_GeMapfunc': s_GeMapfunc.unsqueeze(0),    # (1, 6373)
    's_GeMaplld': s_GeMaplld.reshape(1, -1),    # (1, 65*2997=194805)
    'is_face': is_face,    # (T,)

    'l_AU': l_AU,       # (T, )
    'l_exp': l_exp,     # (T, )
    'l_VA': l_VA,       # (T, )
    'l_mask': l_mask,   # (T, )

    'l_fake_AU': l_fake_AU,     # (T, )
    'l_fake_exp': l_fake_exp,   # (T, )
    'l_fake_VA': l_fake_VA,     # (T, )
}
```

3. The transformer model takes as input **speaker features** rather than images or waves, and predict **tokens of listener reactions**. The predicted tokens **require decoding** before calculate metrics. The detailed structure and forwarding code see [base_transformer.py](src/models/base_transformer.py) and [proposed_model.py](src/models/proposed_model.py).

The pre-processing pipeline:
1. Facial feature extraction:
   1. Detect the face in each frame. We use face detection methods provided by Dlib and OpenCV in this step. Specifically, we first use OpenCV for face detection. If a face is detected, we continue to detect the next frame. If no face is detected, we use Dlib for face detection. If no face is detected after using Dlib, we return that this frame has no face. When either the OpenCV or Dlib method detect a face, we return the coordinates and size information of the face position for cropping.
   
   2. Crop the face if detected, otherwise mark the frame as faceless (usually due to obstacle or extreme orientation). Specifically, we crop the detected faces to a size of 128*128. For frames where no face is detected, we replace them with a zero matrix of the same size.

   3. Extract AU, VA, and expression features from the cropped frame if detected, otherwise skip the frame. In this step, we use [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU) for AU feature extraction, [ResMaskNet](https://github.com/phamquiluan/ResidualMaskingNetwork) for Exp feature extraction, and [Facetorch](https://github.com/tomas-gajarsky/facetorch) for VA feature extraction.

2. Audio feature extraction:
   1. Extract MFCC features with **window size 40ms** and **stride 40ms**.
   2. Extract GeMAPS functional and LLD features of the entire clip by [OpenSmile](https://github.com/audeering/opensmile-python)
   
3. Pose feature extraction:
   1. Save the last layer feature of [VideoMAE](https://github.com/MCG-NJU/VideoMAE) with **window size 5 frames**.

We have provided an all-in-one automatic script for pre-processing, please follow [Data preparation](#data-preparation)


## Installation
**CUDA requirements: CUDA 11.7 and CuDNN 8.5.0.0** (May work for CUDA 11.x and CuDNN 8.x)

1. Download the folder [REACT23_code](https://drive.google.com/drive/folders/1p0J-KhjL3andDawBoTdmLOdB9m-zxYg-?usp=drive_link). 
2. Create a conda environment named "react" and activate it:

```
mkdir -p react
tar -xzf react.tar.gz -C react
source react/bin/activate
conda-unpack
```

3. Uninstall dlib and reinstall it:

```
pip install --force-reinstall dlib
```

4. Extract external scripts and install python_speech_features:

```
cd ./scripts
tar -xzf reaction_to_3DMM.tar.gz
cd ./external
tar -xzf externals.tar.gz
cd ./python_speech_features
python setup.py develop
cd ../../..
```

5. Extract auxiliary models    
For label tokenization.
```cmd
cd ./data/
tar -xzf Index.tar.gz
tar -xzf neighbour_emotions.tar.gz
cd ..
```

## Data Preparation
1. Place the data under `./data` following this structure:
The `.mp4` file should be uncropped video with audio track. 
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1.mp4
                   ├── 2.mp4
                   ├── ....
           ├── ....
       ├── RECOLA
       ├── UDIVA
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
       ├── UDIVA
```

2. Run in the project root:
```cmd
cd scripts
python ./Data_Script.py -d test
```
Command line args:
* **-d** `train`, `val`, or `test`. Default to test.

3. Waiting for the h5 to be constructed.

## Training:
In the project root.

### Pretrain
By default, the training scripts are designed for 4*24G GPUs. You can adjust the hyperparameters based on your machine.
```cmd
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=14900 train.py --config ./configs/pretrain-cos-keeplr.yaml
```

### Finetune
By default, the finetune scripts are designed for 2*24G GPUs. You can adjust the hyperparameters based on your machine.
```cmd
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=14900 train.py --config ./configs/finetune-cos-keeplr.yaml
```

## Evaluation:
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 120000 --config ./configs/cos-keeplr.yaml -t offline -p -m -hf -s
```
Command line args:
* **CUDA_VISIBLE_DEVICES** choose a GPU.
* **--config** a `yaml` file. See ./configs for examples.
* **-i** the iter of model to load
* **-t** task, `online` or `offline`. Default to offline.
* **-d** dataset, `val` or `test`. Default to val.
* **-p** run model prediction and save tokens to `*_tokens_pred.npy`. Roughly 5.5/xx hours for online/offline task.
* **-m** load the `*_tokens_pred.npy` and calculate metrics. If run `-p` and `-m` with the same yaml, the loading path is consistent with the saving path.
* **-s** if True, save decoded results to `./results/`.
* **-hf** reference in FP16 precision.

### Results on the val set
Due to time limit, we have not finished generating 3D/2D videos for the whole validation, thus we cannot report the **FRRea** score.

**Offline Task** 

| model name        | config                          | step | temp          | top-k | top-p       | FRCorr   | FRDist     | FRDiv    | FRVar    | FRDvs    | FRSyn     |  
|-------------------|---------------------------------|------|---------------|-------|-------------|----------|------------|----------|----------|----------|-----------|
| finetune-cos-100k | test-finetune-cos-keeplr-0.yaml | 1    | 0.33          | 5     | 1           | 0.213486 | 93.554599  | 0.05798  | 0.057007 | 0.199116 | 48.781883 |
| finetune-cos-100k | test-finetune-cos-keeplr-1.yaml | 1    | 0.33          | 6     | 1           | 0.216819 | 92.699827  | 0.059412 | 0.056588 | 0.196448 | 48.74255  |
| finetune-cos-100k | test-finetune-cos-keeplr-2.yaml | 1    | 0.5           | 8     | 1           | 0.200294 | 92.728297  | 0.082844 | 0.061665 | 0.200131 | 48.528606 |
| cos-keeplr-150k   | test-cos-keeplr-0.yaml          | 8    | 0.912-0.08[1] | 5     | 0.05+0.1[1] | 0.259561 | 102.800089 | 0.021097 | 0.046578 | 0.165070 | 49.000000 |
[1] `a+b` means we use different parameters to generate the 10 reactions. For each input speaker, the `i`th (from 0 to 9) reaction is generated by param `a + i*b`.

**Online Task** 

| model name       | config                  | window | temp | top-k | top-p | FRCorr   | FRDist     | FRDiv    | FRVar    | FRDvs    | FRSyn     |  
|------------------|-------------------------|--------|------|-------|-------|----------|------------|----------|----------|----------|-----------|
| 5s-randL-gp-150k | test-5s-randL-gp-0.yaml | 5      | 1    | 0     | 0.33  | 0.198886 | 102.552788 | 0.053792 | 0.06612  | 0.103502 | 49.0      |
| 5s-randL-gp-150k | test-5s-randL-gp-1.yaml | 5      | 0.33 | 5     | 1     | 0.208453 | 103.636494 | 0.074614 | 0.065093 | 0.105629 | 49.0      |
| cos-keeplr-150k  | test-cos-keeplr-1.yaml  | 1      | 1    | 5     | 0.33  | 0.173723 | 125.744884 | 0.134123 | 0.091884 | 0.230472 | 47.940929 |

### Commands to generate the submitted test results
Results are saved in `./results`
#### Offline
test_cos-keeplr_150000_offline_s8_t0.912-0.08_k5_p0.05+0.1_pred.npy (3 hours)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 150000 --config ./configs/test-cos-keeplr-0.yaml -t offline -d test -p -hf
python src/util.py -d ./ckpts/pretrain-cos-klr/test_result/test_pretrain-cos-klr_150000_offline_s8_t0.912-0.08_k5_p0.05+0.1_tokens_pred.npy
```


test_finetune_100k_offline_s1_t0.33_k5_p1_pred.npy (half an hour)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 100000 --config ./configs/test-finetune-cos-keeplr-0.yaml -t offline -d test -p -hf
python src/util.py -d ./ckpts/finetune-cos-klr/test_result/test_finetune-cos-klr_100000_offline_s1_t0.33_k5_p1_tokens_pred.npy
```


test_finetune_100000_offline_s1_t0.33_k6_p1_pred.npy (half an hour)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 100000 --config ./configs/test-finetune-cos-keeplr-1.yaml -t offline -d test -p -hf
python src/util.py -d ./ckpts/finetune-cos-klr/test_result/test_finetune-cos-klr_100000_offline_s1_t0.33_k6_p1_tokens_pred.npy
```


test_finetune_100000_offline_s1_t0.5_k8_p1_pred.npy (half an hour)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 100000 --config ./configs/test-finetune-cos-keeplr-2.yaml -t offline -d test -p -hf
python src/util.py -d ./ckpts/finetune-cos-klr/test_result/test_finetune-cos-klr_100000_offline_s1_t0.5_k8_p1_tokens_pred.npy
```

#### Online
test_cos-klr_150000_online_s1_t1.0_k5_p0.33_pred.npy (1 day)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 150000 --config ./configs/test-cos-keeplr-1.yaml -t online -d test -p -hf
python src/util.py -d ./ckpts/pretrain-cos-klr/test_result/test_pretrain-cos-klr_150000_online_s1_t1.0_k5_p0.33_tokens_pred.npy
```


test_5s-randL-gp_online_s5_t1.0_k0_p0.33_pred.npy (1 day)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 150000 --config ./configs/test-5s-randL-gp-0.yaml -t online -d test -p -hf
python src/util.py -d ./ckpts/pretrain-5s-randL-gp/test_result/test_pretrain-5s-randL-gp_150000_online_s5_t1.0_k0_p0.33_tokens_pred.npy
```


test_5s-randL-gp_150000_online_s5_t0.33_k5_p1_pred.npy (1 day)
```cmd
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=14900 test.py -i 150000 --config ./configs/test-5s-randL-gp-1.yaml -t online -d test -p -hf
python src/util.py -d ./ckpts/pretrain-5s-randL-gp/test_result/test_pretrain-5s-randL-gp_150000_online_s5_t0.33_k5_p1_tokens_pred.npy
```



## Visualization
In the project root.

We separately trained a 2-layer Bi-LSTM for mapping from facial reactions (AUs, VA, and expressions) to 3DMM coefficients.  
Train the Bi-LSTM:   
```cmd
python scripts/reaction_to_3DMM/lstmtrain.py 
```

After generated decoded results `**_pred.npy` of the val set with the shape `(N, 10, 750, 25)`, run the following command to select the 5 samples for visualization:
```cmd
python src/util.py -s path_of_pred.npy
```
Then, a `**_5_samples.pkl` file will be generated in `results/`.

Extract the decoded emotion label into `**.csv` with the shape(750,25)

```
python scripts/reaction_to_3DMM/read.py --path results/**_5_samples.pkl
```

Predict 3DMM coefficients, `**.npy` with the shape(750,1,1,58)

```cmd
python  scripts/reaction_to_3DMM/lstmtest.py
```

Render  videos into `scripts/reaction_to_3DMM/visual_output`:
```cmd
python  scripts/reaction_to_3DMM/render_all.py
```
Detail of Visualization is in the   `scripts/reaction_to_3DMM/README.md`.

## Pretrained Models

We provide checkpoints in the `./ckpts`.
```cmd
cd ckpts
tar -xzf pretrain-5s-randL-gp.tar.gz
tar -xzf pretrain-cos-klr.tar.gz
tar -xzf finetune-cos-klr.tar.gz
cd ..
```

## Credits:

This repo is adapted from https://github.com/xrenaa/Look-Outside-Room.

Attention Layers and train/inference are based on https://github.com/karpathy/minGPT and https://github.com/google-research/maskgit.
