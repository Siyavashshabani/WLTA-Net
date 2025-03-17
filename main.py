import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from train.utils import train
from model.cross_attention.model import WLTANet
from loader.loader import data_loaders

def main():
    # Load hyperparameters from JSON file
    with open('configs.json', 'r') as config_file:
        config = json.load(config_file)

    # Set device to use all available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    val_loader, train_loader = data_loaders(config["data_dir"], num_samples=config["num_samples"], device=device)

    # define the model
    model = WLTANet(config, 
        img_size=(config["input_size"], config["input_size"], config["input_size"]),
        in_channels=config["input_channels"],
        out_channels=config["num_classes"],
        feature_size=config["feature_size"],
        use_checkpoint=False, 
        n_patches =config["n_patches"],
        input_dim_0=96, n_channels_0=1  ,
        input_dim_1=48, n_channels_1=48 ,
        input_dim_2=24, n_channels_2=96 ,
        input_dim_3=12, n_channels_3=192,
        input_dim_4=6 , n_channels_4=384,
        )

    # Wrap the model with DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    # train loop 
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    while global_step < config["max_iterations"]:
        global_step, dice_val_best, global_step_best = train(
            model, global_step, train_loader, val_loader, config, dice_val_best, global_step_best
        )

if __name__ == "__main__":
    main()
