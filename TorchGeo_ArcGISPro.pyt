# -*- coding: utf-8 -*-

import arcpy
import os
import glob
import torch
import rasterio
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

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
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions."""
        param0 = arcpy.Parameter(
            displayName="Input Image Layer",
            name="in_image",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            multiValue=True,
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
        image_layers = parameters[0].values  # List of input raster layers
        mask_layer = parameters[1].value
        out_folder = parameters[2].valueAsText
        batch_size = parameters[3].value
        epochs = parameters[4].value
        size = 64

        os.makedirs(out_folder, exist_ok=True)

        # Initialize RasterDataset for input images
        image_dataset = RasterDataset(paths=[arcpy.Describe(layer).catalogPath for layer in image_layers])
        messages.addMessage(f"Image Dataset: {image_dataset}")

        # Initialize VectorDataset for mask layer
        mask_path = arcpy.Describe(mask_layer).catalogPath
        mask_dataset = RasterDataset(paths=[mask_path])
        messages.addMessage(f"Mask Dataset: {mask_dataset}")

        # Combine image and mask datasets
        dataset = image_dataset & mask_dataset
        messages.addMessage(f"Intersection CRS: {dataset.crs}")

        # Configure sampler and dataloader
        sampler = GridGeoSampler(dataset, size=size, stride=size)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples)

        # Determine number of classes from mask dataset
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            num_classes = len(np.unique(mask_data))

        # Configure and train the model
        messages.addMessage("Configuring the model...")
        task = SemanticSegmentationTask(
            model="unet",
            backbone="resnet50",
            in_channels=1,  # Assuming 1 band for simplicity
            num_classes=num_classes,
            loss="ce",
            lr=1e-3,
        )

        # Configure logger
        logger = TensorBoardLogger(save_dir=out_folder, name="segmentation_logs")

        # Configure trainer
        trainer = Trainer(
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=10,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

        # Start training
        messages.addMessage("Starting the training process...")
        trainer.fit(model=task, dataloader=dataloader)
        messages.addMessage("Training completed successfully.")
