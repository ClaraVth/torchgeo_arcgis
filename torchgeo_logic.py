import torch
from torch import Tensor
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from typing import Dict, Optional

import rasterio
import numpy as np
import os

# ------------------------- Helper Classes -------------------------
class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def transfer_batch_to_device(self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int) -> Dict[str, Tensor]:
        """Transfer batch to device.

        Removes non-Tensor data (like `crs` and `bbox`) and sends Tensor data to the device.

        Args:
            batch (dict): The batch to transfer.
            device (torch.device): The target device.
            dataloader_idx (int): Index of the DataLoader.

        Returns:
            dict: The batch with Tensor data transferred to the device.
        """
        if "crs" in batch:
            del batch["crs"]
        if "bbox" in batch:
            del batch["bbox"]
        return super().transfer_batch_to_device(batch, device, dataloader_idx)
    

class CostumGeoDataModule(LightningDataModule):
    def __init__(self, image_path: str, mask_path: str, patch_size: int, batch_size: int, num_workers: int = 0):
        """
        Initialize the data module.

        Args:
            image_path (str): Path to the image dataset.
            mask_path (str): Path to the mask dataset.
            patch_size (int): Size of each patch.
            batch_size (int): Size of each mini-batch.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None  # Combined dataset

    def setup(self, stage: Optional[str] = None):
        """Prepare datasets for training and validation."""
        # Initialize RasterDatasets
        image_dataset = RasterDataset(paths=self.image_path)
        mask_dataset = RasterDataset(paths=self.mask_path)

        # Combine image and mask datasets
        self.dataset = image_dataset & mask_dataset

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for training."""
        sampler = GridGeoSampler(self.dataset, size=self.patch_size, stride=self.patch_size)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=stack_samples,
            num_workers=self.num_workers,
        )

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
    #dataset = image_dataset & mask_dataset

    data_module = CostumGeoDataModule(image_path, mask_path, patch_size, batch_size)

    # Configure Sampler and DataLoader
    #sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
    #dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples)

    # Extract number of classes from mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        num_classes = len(np.unique(mask_data))

    # Configure the model
    task = CustomSemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        in_channels=3,  # Assumption: 3 channels in input image
        num_classes=num_classes,
        loss="ce",
        lr=1e-3,
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
    trainer.fit(model=task, datamodule=data_module)

in_image = r"data\LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF"
in_mask = r"data\2023_30m_cdls.tif"
out_folder = r"data"
batch_size = 8
epochs = 10

train_model(in_image, in_mask, out_folder, batch_size, epochs)