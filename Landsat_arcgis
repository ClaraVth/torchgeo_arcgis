# -*- coding: utf-8 -*-

import arcpy
import os
import glob
import torch
import rasterio
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import CDL, Landsat7, Landsat8, stack_samples
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

        # Dropdown menu parameter
        param01 = arcpy.Parameter(
            displayName="Select Dataset Source",
            name="source_dataset",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
            multiValue=True,
        )
        
        # Define dropdown options
        param01.filter.type = "ValueList"
        param01.filter.list = [
            "Landsat7",
            "Landsat8",
        ]

        param1 = arcpy.Parameter(
            displayName="Input Mask Layer",
            name="in_mask",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
        )

        # Dropdown menu parameter
        param11 = arcpy.Parameter(
            displayName="Select Mask Type",
            name="mask_type",
            datatype="GPString",
            parameterType="Required",
            direction="Input",
        )
        
        # Define dropdown options
        param11.filter.type = "ValueList"
        param11.filter.list = [
            "CDL"
        ]

        param2 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Output",
        )
        param3 = arcpy.Parameter(
            displayName="Batch Size",
            name="batch_size",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        param3.value = 8 # collapsed with 16

        param4 = arcpy.Parameter(
            displayName="Number of Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        param4.value = 10

        return [param0, param01, param1, param11, param2, param3, param4]

    def execute(self, parameters, messages):
        """Main execution logic of the tool."""
        image_layers = parameters[0].values  # List of input raster layers
        selected_sources = parameters[1].values  # List of selected datasets
        mask_layer = parameters[2].value
        mask_type = parameters[3].valueAsText
        out_folder = parameters[4].valueAsText
        batch_size = parameters[5].value
        size = 64 # collapsed with 128

        landsat_root = os.path.join(out_folder, "landsat")
        os.makedirs(landsat_root, exist_ok=True)
        mask_root = os.path.join(out_folder, 'Mask')
        os.makedirs(mask_root, exist_ok=True)

        # Create subdirectories for Landsat7 and Landsat8
        landsat7_dir = os.path.join(landsat_root, "Landsat7")
        landsat8_dir = os.path.join(landsat_root, "Landsat8")
        os.makedirs(landsat7_dir, exist_ok=True)
        os.makedirs(landsat8_dir, exist_ok=True)
        
        landsat7_bands = []
        landsat8_bands = []

        for layer in image_layers:
            layer_path = arcpy.Describe(layer).catalogPath
            layer_name = os.path.basename(layer.dataSource)
            
            # Debugging
            #crs_info = arcpy.Describe(layer).spatialReference.name
            #messages.addMessage(f"Layer: {layer_name}, CRS: {crs_info}")
            
            if "LE07" in layer_name and "Landsat7" in selected_sources:
                band = "_".join(layer_name.split("_")[-2:]).split(".")[0]
                landsat7_bands.append(band)
                destination = os.path.join(landsat7_dir, layer_name)
                arcpy.management.CopyRaster(layer_path, destination)

            elif "LC08" in layer_name and "Landsat8" in selected_sources:
                band = "_".join(layer_name.split("_")[-2:]).split(".")[0]
                landsat8_bands.append(band)
                destination = os.path.join(landsat8_dir, layer_name)
                arcpy.management.CopyRaster(layer_path, destination)

        # Print or log the selected bands for debugging
        #messages.addMessage(f"Landsat 7 Bands: {landsat7_bands}")
        #messages.addMessage(f"Landsat 8 Bands: {landsat8_bands}")

        mask_path = arcpy.Describe(mask_layer).catalogPath
        mask_name = os.path.basename(mask_layer.dataSource)
        mask_destination = os.path.join(mask_root, mask_name)
        arcpy.management.CopyRaster(mask_path, mask_destination)

        # Find .ovr files - ArcGIS specific
        ovr_files = glob.glob(os.path.join(out_folder, "**", "*.ovr"), recursive=True)

        # Remove .ovr-files
        for ovr_file in ovr_files:
            os.remove(ovr_file)

        
        # Initialize Landsat datasets
        if landsat7_bands:
            landsat7 = Landsat7(paths=landsat7_dir, bands=landsat7_bands)
            messages.addMessage(f"Landsat 7 Dataset: {landsat7}")
            messages.addMessage(f"Landsat 7 CRS: {landsat7.crs}")

        if landsat8_bands:
            landsat8 = Landsat8(paths=landsat8_dir, bands=landsat8_bands)
            messages.addMessage(f"Landsat 8 Dataset: {landsat8}")
            messages.addMessage(f"Landsat 8 CRS: {landsat8.crs}")

        # Initialise Mask dataset
        if mask_type == "CDL":
            cdl = CDL(paths=mask_root)
            messages.addMessage(f"Mask Dataset: {cdl}")
            messages.addMessage(f"CDL CRS: {cdl.crs}")
        
        if "Landsat7" in selected_sources and "Landsat8" in selected_sources:
            landsat = landsat7 | landsat8
        
        dataset = landsat & cdl
        messages.addMessage(f"Intersection CRS: {dataset.crs}")

        test_sampler = GridGeoSampler(dataset, size=size, stride=size)
        next(iter(test_sampler))

        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=stack_samples)


        with rasterio.open(mask_destination) as src:
            data = src.read(1)
            num_classes_layer = len(np.unique(data))
            pass     


        # Configure and train the model
        messages.addMessage("Configuring the model...")
        task = SemanticSegmentationTask(
            model="unet",
            backbone="resnet50",
            in_channels=max(len(landsat7_bands), len(landsat8_bands)),
            num_classes=num_classes_layer,  # Adjust based on your dataset
            loss="ce",
            lr=1e-3,
        )

        # Configure logger
        logger = TensorBoardLogger(save_dir=out_folder, name="segmentation_logs")

        # Configure trainer
        trainer = Trainer(
            max_epochs=parameters[6].value,  # Use the number of epochs from the input parameter
            logger=logger,
            log_every_n_steps=10,  # Log every 10 steps
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

        # Start training
        messages.addMessage("Starting the training process...")
        trainer.fit(task, train_dataloaders=test_dataloader)
        messages.addMessage("Training completed successfully.")

        return