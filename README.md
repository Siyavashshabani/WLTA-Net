
This is the official PyTorch implementation of the paper - "A Novel 3D Decoder with Weighted and Learnable Triple Attention for 3D Microscopy Image Segmentation". for Workshops on Computer Vision for Microscopy Image Analysis on CVPR Conference 2025

## A Novel 3D Decoder with Weighted and Learnable Triple Attention for 3D Microscopy Image Segmentation

![](diagram/model.png)

## Set Up Set Up Environment
```
python3.10 -m venv ViT_env 
source CSTANet_env/bin/activate 
pip install -r requirements.txt
```
## Datasets

Download synapse dataset here: https://www.synapse.org/#!Synapse:syn3193805/wiki/89480

Download ACDC dataset here: https://www.creatis.insa-lyon.fr/Challenge/acdc/


### Structure of synapse dataset folders 

data/  
&nbsp;|---imagesTr/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---img0001.nii.gz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---img0002.nii.gz  
&nbsp;|---labelsTr/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---label0001.nii.gz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---label0002.nii.gz  
&nbsp;|---dataset.json  


## Running the Code


## Training

Before training the configs.json file should be filled:

### Hyperparameters Configuration
The following settings can be adjusted in the `config.py` to configure the model training and data management:

- **`data_dir`**: Set the directory path for dataset storage.
- **`saved_model_dir`**: Set the directory path where trained models and checkpoints will be saved.
- **`num_samples`**: Define the number of samples used in training process.
- **`num_classes`**: Specify the number of target classes in the dataset + background.
- **`input_size`**: Set the size of the input images or data.
- **`input_channels`**: Define the number of input channels for the data (e.g., grayscale=1, RGB=3).
- **`feature_size`**: Set the dimension of the feature vectors extracted by the model.
- **`use_checkpoint`**: Enable or disable the use of model checkpoints for training continuation.
- **`learning_rate`**: Set the initial learning rate for the optimizer.
- **`weight_decay`**: Define the weight decay (L2 penalty) rate for regularization.
- **`max_iterations`**: Specify the maximum number of iterations (or training steps).
- **`eval_num`**: Set the frequency of evaluations (every `eval_num` iterations) during training.

- **'type_combine'**: Set the combination of Triple Attention
## train the model 

```
python3.10 main.py  
```
<!-- ## Results

### Quantitative
![](diagram/synapse_num.jpg)

### Visulization 
## Synapse 
![](diagram/synapse_vis.jpg)

## ACDC
![](diagram/acdc_vis.png) -->

## Citation
If any part of this code is used, please give appropriate citations to our paper. <br />

## Contact
