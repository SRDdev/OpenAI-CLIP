import torch
import pandas as pd
import numpy as np
import config as config
from dataset import get_transforms,CLIPDataset
from utils import *
from tqdm import tqdm
from transformers import AutoTokenizer
from CLIP import *
import itertools

#----------------------Dataset--------------------#
def make_train_valid_dfs():
    """
    Define the function to split the dataset into training and validation sets.
    Args:
        Inputs: None
        Outputs:
            train_dataframe: DataFrame for training
            valid_dataframe: DataFrame for validation
    Notes: The split ratio is determined by the 'config.debug' variable.
    """
    dataframe = pd.read_csv(f"{config.captions_path}/captions.csv")

    # Add 'id' column if not present
    if 'id' not in dataframe.columns:
        dataframe['id'] = range(len(dataframe))

    max_id = dataframe["id"].max() + 1 if not config.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def build_dataloaders(dataframe,tokenizer,mode):
    """
    Define the function to create data loaders for training and validation.
    Args:
        Inputs:
            dataframe: DataFrame containing image captions
            tokenizer: Tokenizer for processing text
            mode: String specifying 'train' or 'valid' mode
        Outputs:
            dataloader: PyTorch DataLoader for the specified mode
    Notes: Data augmentation is applied in training mode.
    """
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    """
    Define the function to perform one training epoch.
    Args:
        Inputs:
            model: CLIPModel instance
            train_loader: DataLoader for training data
            optimizer: PyTorch optimizer
            lr_scheduler: Learning rate scheduler
            step: String specifying 'epoch' or 'batch' for LR scheduling
        Outputs:
            loss_meter: AverageMeter instance containing training loss
    Notes: LR scheduling is applied based on the specified step.
    """
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(config.device) for k,v in batch.items() if k!="caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step=="batch":
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        return loss_meter
    
def valid_epoch(model, valid_loader):
    """
    Define the function to perform one validation epoch.
    Args:
        Inputs:
            model: CLIPModel instance
            valid_loader: DataLoader for validation data
        Outputs:
            loss_meter: AverageMeter instance containing validation loss
    Notes: No backpropagation is performed during validation.
    """
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(config.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def trainer():
    """
    Define the main training function.
    Args: None
    Outputs: None
    Notes: Training loop with LR scheduling and model saving based on validation loss.
    """
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = AutoTokenizer.from_pretrained(config.text_tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    train_loader = build_dataloaders(train_df, tokenizer, mode="train")
    valid_loader = build_dataloaders(valid_df, tokenizer, mode="valid")
    print(f"Deivce : {config.device}")
    model = CLIPModel().to(config.device)

    params = [
        {"params": model.image_encoder.parameters(), "lr": config.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": config.text_encoder_lr},
        {"params": itertools.chain(model.image_proj.parameters(),
                                model.text_proj.parameters()),
        "lr": config.head_lr, "weight_decay": config.weight_decay
        }
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.patience, factor=config.factor
    )
    step = "epoch"
    best_loss = float('inf')
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "../models/clip_model.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    trainer()