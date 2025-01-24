import torch
from torch import Tensor
from torchgeo.datasets import RasterDataset, stack_samples, GeoDataset
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
class DropFrozenKeys:
    def __call__(self, sample):
        sample.pop('crs')
        bbox = sample['bounds']
        sample['bounds'] = torch.tensor([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
        #sample['bounds']=torch.tensor(sample['bounds'])
        return sample

# ------------------------- Tool Definition -------------------------

def train_model(image_path, mask_path, out_folder, batch_size, epochs, patch_size=64):
    """
    Führt die gesamte Trainingslogik für das TorchGeo-Modell aus.

    Args:
        image_path (str): Path to input image.
        mask_path (str): Path to the mask.
        out_folder (str): Output folder.
        batch_size (int): Batch size.
        epochs (int): Number of Training epochs.
        patch_size (int): Path size for the model.

    Returns:
        None
    """
    os.makedirs(out_folder, exist_ok=True)

    transforms = DropFrozenKeys()

    # Initialize datasets
    image_dataset = RasterDataset(paths=image_path, transforms=transforms)
    mask_dataset = RasterDataset(paths=mask_path, transforms=transforms)
    mask_dataset.is_image = False

    # combine datasets
    dataset = image_dataset & mask_dataset
    print(dataset)

    

    # Configure Sampler and DataLoader
    sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=stack_samples)
    for batch in dataloader:
        x, y = batch["image"], batch["mask"]
        #y = y.squeeze(1)  # Remove channel dimension
        #x = x.squeeze(1)
        print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        break


    # Extract number of classes from mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        num_classes = len(np.unique(mask_data))

    # Configure the model
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        in_channels=3,  # Assumption: 1 channel in input image
        num_classes=num_classes,
        loss="ce",
        lr=1e-3,
    )

    """
    data_module = GeoDataModule(
        dataset_class=RasterDataset,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=4,
        paths=[image_path, mask_path],
    )
    """
    
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
    print("start training ...")
    trainer.fit(model=task, train_dataloaders=dataloader) #, datamodule=data_module

"""
in_files = ["data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B2.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B3.TIF"]
#out_file = "data/multispectral_image.TIF"

with rasterio.open(in_files[0]) as src:
    metadata = src.meta
metadata.update({
    "count": len(in_files),  # Number of bands
})
with rasterio.open(out_file, "w", **metadata) as dst:
    for band_index, file in enumerate(in_files, start=1):
        with rasterio.open(file) as src:
            band = src.read(1)  # Read first band
            dst.write(band, band_index)  # Write band in corresponding position

print(f"Multispectral image: {out_file}")
"""
in_image = "data/multispectral_image.TIF"
in_mask = "data/2023_30m_cdls.tif"
out_folder = "."
batch_size = 8
epochs = 10

train_model(in_image, in_mask, out_folder, batch_size, epochs)
