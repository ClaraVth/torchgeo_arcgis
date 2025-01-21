
import torch
from torch import Tensor
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from torchgeo.datamodules import GeoDataModule
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

import rasterio
import numpy as np
import os

# ------------------------- Helper Classes -------------------------
class CustomGeoDataModule(GeoDataModule):
    def __init__(self, dataset, batch_size=1, patch_size=64, num_workers=0):
        super().__init__(
            dataset_class=None,  # Kein `dataset_class` verwenden
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
        )
        self.dataset = dataset  # Speichern des kombinierten Datasets

    def setup(self, stage: Optional[str] = None):
        """Set up datasets and define samplers."""
        if not hasattr(self, "dataset") or self.dataset is None:
            raise ValueError("Dataset must be provided to CustomGeoDataModule!")

        # Sampler für das Training und die Validierung konfigurieren
        if stage in ("fit", "validate"):
            self.train_sampler = GridGeoSampler(
                self.dataset, size=self.patch_size, stride=self.patch_size
            )
            self.val_sampler = GridGeoSampler(
                self.dataset, size=self.patch_size, stride=self.patch_size
            )

        # Falls notwendig, können weitere Sampler für Test oder Prediction hinzugefügt werden

    def train_dataloader(self) -> DataLoader:
        """Definiert den DataLoader für das Training."""
        if self.train_sampler is None:
            raise MisconfigurationException(
                "train_sampler must be defined in CustomGeoDataModule.setup"
            )
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader:
        """Definiert den DataLoader für die Validierung."""
        if self.val_sampler is None:
            raise MisconfigurationException(
                "val_sampler must be defined in CustomGeoDataModule.setup"
            )
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
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
    dataset = image_dataset & mask_dataset

    #data_module = CostumGeoDataModule(image_path, mask_path, patch_size, batch_size)

    # Configure Sampler and DataLoader
    sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
    #dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples)

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

    data_module = CustomGeoDataModule(
        dataset = dataset,
        #dataset_class=type[dataset],
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=4,
        #paths={"image": image_path, "mask": mask_path},
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

in_image = r"data\LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.tif"
in_mask = r"data\2023_30m_cdls.tif"
out_folder = r"."
batch_size = 8
epochs = 10

with rasterio.open(in_image) as src:
    print(src.meta)

with rasterio.open(in_mask) as src:
    print(src.meta)

train_model(in_image, in_mask, out_folder, batch_size, epochs)