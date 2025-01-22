# -*- coding: utf-8 -*-

import arcpy
import os
import glob
import torch
import rasterio
import numpy as np
import gc
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples

from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

# ------------------------- Helper functions -------------------------
class DropFrozenKeys:
    def __call__(self, sample):
        sample.pop('crs')
        sample.pop('bounds')
        return sample

# ------------------------- Toolbox Definition -------------------------
class Toolbox:
    def __init__(self):
        """ArcGIS Python Toolbox for training land use classification models using TorchGeo."""
        self.label = "TorchGeo Land Use Classification Toolbox"
        self.alias = "torchgeo_landuse"
        self.tools = [TrainLandUseModel]

# ------------------------- Tool Definition -------------------------
class TrainLandUseModel:
    def __init__(self):
        self.label = "Train Land Cover Mapping Model"
        self.description = "Train a TorchGeo-based semantic segmentation model."
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions."""
        param0 = arcpy.Parameter(
            displayName="Input Image Layer",
            name="in_image",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            #multiValue=True,
        )

        param01 = arcpy.Parameter(
            displayName="Input Mask Layer",
            name="in_mask",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        param1 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Output",
        )
        param2 = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        param2.value = 8

        param3 = arcpy.Parameter(
            displayName="Number of Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        param3.value = 10

        return [param0, param01, param1, param2, param3]

    def execute(self, parameters, messages):
        """Main execution logic of the tool."""
        gc.collect()
        torch.cuda.empty_cache()
        
        #image_layers = parameters[0].values  # List of input raster layers
        image_layers = parameters[0].value  # For only 1 possible input
        mask_layer = parameters[1].value
        out_folder = parameters[2].valueAsText
        batch_size = parameters[3].value
        epochs = parameters[4].value
        patch_size = 64

        os.makedirs(out_folder, exist_ok=True)

        transforms = DropFrozenKeys()

        # Initialize datasets
        image_dataset = RasterDataset(paths=arcpy.Describe(image_layers).catalogPath, transforms=transforms)
        messages.addMessage(f"Image Dataset: {image_dataset}")
        mask_path = arcpy.Describe(mask_layer).catalogPath
        mask_dataset = RasterDataset(paths=[mask_path], transforms=transforms)
        messages.addMessage(f"Mask Dataset: {mask_dataset}")
        mask_dataset.is_image = False

        # Determine number of classes from mask dataset
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            num_classes = len(np.unique(mask_data))

        # combine datasets
        dataset = image_dataset & mask_dataset
        messages.addMessage(f"Intersection CRS: {dataset.crs}")

        #data_module = CostumGeoDataModule(image_path, mask_path, patch_size, batch_size)

        # Configure Sampler and DataLoader
        sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=stack_samples)
        for batch in dataloader:
            x, y = batch["image"], batch["mask"]
            y = y.squeeze(1)  # Remove channel dimension
            x = x.squeeze(1)
            print(f"x.shape: {x.shape}, y.shape: {y.shape}")
            break


        # Extract number of classes from mask
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            num_classes = len(np.unique(mask_data))



        # Configure the model
        messages.addMessage("Configuring the model...")
        task = SemanticSegmentationTask(
            model="unet",
            backbone="resnet50",
            in_channels=1,  # Assumption: 1 channel in input image
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
            accelerator="gpu" if torch.cuda.is_available() else "cpu",  # if torch.cuda.is_available() else "cpu",
        )

        # Start training
        messages.addMessage("Starting the training process...")
        trainer.fit(model=task, train_dataloaders=dataloader) #, datamodule=data_module
        messages.addMessage("Training completed successfully.")

