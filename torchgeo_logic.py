import torch
from torch import Tensor
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from torchgeo.datamodules import GeoDataModule
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

import rasterio
import numpy as np
import os

# ------------------------- Helper Classes -------------------------


# ------------------------- Tool Definition -------------------------

def train_model(image_path, mask_path, out_folder, batch_size, epochs, patch_size=64):
    """
    Führt die gesamte Trainingslogik für das TorchGeo-Modell aus.

    Args:
        image_path (str): Pfad zur Eingabedatei (Rasterbild).
        mask_path (str): Pfad zur Maskendatei.
        out_folder (str): Ausgabeverzeichnis.
        batch_size (int): Batch-Größe.
        epochs (int): Anzahl der Trainings-Epochen.
        patch_size (int): Größe der Patches für das Modell.

    Returns:
        None
    """
    os.makedirs(out_folder, exist_ok=True)

    # Initialize datasets
    image_dataset = RasterDataset(paths=image_path)
    mask_dataset = RasterDataset(paths=mask_path)

    # combine datasets
    dataset = image_dataset & mask_dataset

    #data_module = CostumGeoDataModule(image_path, mask_path, patch_size, batch_size)

    # Configure Sampler and DataLoader
    sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples)

    # Extract number of classes from mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        num_classes = len(np.unique(mask_data))

    # Configure the model
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        in_channels=3,  # Assumption: 3 channels in input image
        num_classes=num_classes,
        loss="ce",
        lr=1e-3,
    )

    data_module = GeoDataModule(
        dataset_class=RasterDataset,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=4,
        paths=[image_path, mask_path],
    )

    # Configure Logger
    logger = TensorBoardLogger(save_dir=out_folder, name="segmentation_logs")

    # Configure Trainer 
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    trainer.fit(model=task, datamodule=data_module) #, train_dataloaders=dataloader

in_image = r"data\LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF"
in_mask = r"data\2023_30m_cdls.tif"
out_folder = r"."
batch_size = 8
epochs = 10

train_model(in_image, in_mask, out_folder, batch_size, epochs)