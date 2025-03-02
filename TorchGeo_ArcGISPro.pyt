# -*- coding: utf-8 -*-

import arcpy
import os
import glob
import torch
import rasterio
import numpy as np
import gc
from torch import Tensor
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, stack_samples, GeoDataset
from torchgeo.transforms import AugmentationSequential
from torchgeo.models import ResNet50_Weights
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Dict, Optional, Any, Iterable, Mapping

from multiprocessing import Lock
from tqdm import tqdm
import kornia
from kornia.geometry.transform import warp_affine
from tqdm import tqdm

# ------------------------- Helper functions -------------------------
class DropFrozenKeys:
    def __call__(self, sample):
        sample.pop('crs')
        sample.pop('bounds')
        return sample
    
def combine_bands(in_files):
    out_file = "data/multispectral_image.TIF"

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

    #messages.addMessage(f"Multispectral image: {out_file}")
    return(out_file)


def list_dict_to_dict_list(samples: Iterable[Mapping[Any, Any]]) -> dict[Any, Any]:
    """
    Converts a list of dictionaries into a dictionary of lists.

    Args:
        samples: Iterable of dictionaries to be converted.

    Returns:
        A dictionary where each key maps to a list of values from the input dictionaries.
    """
    result = {}
    for sample in samples:
        for key, value in sample.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    return result

def custom_stack_samples(samples: Iterable[Mapping[Any, Any]]) -> dict[Any, Any]:
    """Custom collate function to stack samples and ensure correct dimensions.

    Args:
        samples: List of samples, each being a dictionary with keys like "image" and "mask".

    Returns:
        A single sample (dictionary) with stacked tensors and adjusted dimensions.

    """
    # Convert list of dictionaries to dictionary of lists
    collated: dict[Any, Any] = list_dict_to_dict_list(samples)

    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            # Stack tensors along a new axis
            stacked_value = torch.stack(value)

            # Adjust dimensions for "mask" to remove any unnecessary channel dimension
            if key == "mask":
                stacked_value = stacked_value.squeeze(1)  # Ensure shape is [Batch, Height, Width]

            collated[key] = stacked_value

    return collated

def preprocess_mask(mask_path, out_folder):
    """
    Preprocesses the mask to map unique class values to sequential indices.
    Only keeps classes that represent at least 1% of the dataset.

    Args:
        mask_path (str): Path to the mask image.

    Returns:
        processed_mask_path (str): Path to the processed mask image with indices.
        class_to_index (dict): Mapping from original class values to indices.
        index_to_class (dict): Mapping from indices back to original class values.
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read the first band
        meta = src.meta.copy()

    # Calculate the percentage of each class
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    class_percentages = {cls: count / total_pixels for cls, count in zip(unique, counts)}

    # Filter classes with at least 1% of the dataset
    filtered_classes = {cls for cls, percentage in class_percentages.items() if percentage >= 0.01}
    print(f"Classes with >= 1% of the data: {filtered_classes}")

    # Create mappings for filtered classes
    class_to_index = {}
    current_index = 1

    for cls in unique:
        if cls in filtered_classes:
            class_to_index[cls] = current_index
            current_index += 1
        else:
            class_to_index[cls] = 0  # Assign all other classes to 0

    index_to_class = {idx: cls for cls, idx in class_to_index.items() if idx != 0}
    index_to_class[0] = 0

    # Replace class values with indices
    indexed_mask = np.vectorize(class_to_index.get)(mask)

    # Save the processed mask
    processed_mask_path = os.path.join(out_folder, "processed_mask.tif")
    meta.update({"dtype": "uint8", "count": 1})
    with rasterio.open(processed_mask_path, "w", **meta) as dst:
        dst.write(indexed_mask.astype(np.uint8), 1)

    print(f"Processed mask saved to {processed_mask_path}")
    return processed_mask_path, index_to_class


def postprocess_prediction(prediction_path, output_path, index_to_class):
    """
    Postprocesses the prediction by mapping indices back to original class values.

    Args:
        prediction_path (str): Path to the predicted image with indices.
        output_path (str): Path to save the processed output image.
        index_to_class (dict): Mapping from indices to original class values.

    Returns:
        None
    """
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)  # Read the first band
        meta = src.meta.copy()

    # Replace indices with original class values
    remapped_prediction = np.vectorize(index_to_class.get)(prediction)

    # Save the remapped prediction
    meta.update({"dtype": "uint8", "count": 1})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(remapped_prediction.astype(np.uint8), 1)

    print(f"Postprocessed prediction saved to {output_path}")

# ------------------------- Tool Definitions -------------------------
# ------------------------- Training -------------------------
def train_model(image_path, mask_path, out_folder, batch_size, epochs, num_workers, patch_size):
    """
    Args:
        image_path (str): Path to input image.
        mask_path (str): Path to the mask.
        out_folder (str): Output folder.
        batch_size (int): Batch size.
        epochs (int): Number of Training epochs.
        patch_size (int): Path size for the model.
    """

    drop_frozen_keys = DropFrozenKeys()

    # Data Augmentation
    augmentation_transforms = AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=90),
        #RandomCrop(size=(64, 64)),
        data_keys=["image", "mask"]
    )

    # Initialize datasets
    image_dataset = RasterDataset(paths=image_path, transforms=drop_frozen_keys)
    mask_dataset = RasterDataset(paths=mask_path, transforms=drop_frozen_keys)
    mask_dataset.is_image = False

    # combine datasets
    dataset = image_dataset & mask_dataset

    dataset.transforms = augmentation_transforms

    weights = ResNet50_Weights.LANDSAT_OLI_SR_SIMCLR

    

    # Configure Sampler and DataLoader
    sampler = RandomGeoSampler(dataset, size=patch_size, length=10000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=custom_stack_samples,
        pin_memory=torch.backends.mps.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    #print(f"Dataset keys: {list(dataset[0].keys())}")
    """
    for batch in dataloader:
        x, y = batch["image"], batch["mask"]
        print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        break
    """

    # Extract number of classes from mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        #print(src.read(1))
        num_classes = len(np.unique(mask_data))
        print(f"Number of classes: {num_classes}")
    
    with rasterio.open(image_path) as src:
        num_bands = src.count
        print(f"Number of bands: {num_bands}")

    # Configure the model
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        weights=weights,
        in_channels=num_bands,
        num_classes=num_classes,
        loss="ce",
        lr=1e-3,
    )
    
    # Configure Logger
    logger = TensorBoardLogger(save_dir=out_folder, name="segmentation_logs")

    val_sampler = GridGeoSampler(dataset, size=patch_size, stride=0.5*patch_size)
    val_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=custom_stack_samples,
        pin_memory=torch.backends.mps.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )


    # Configure Trainer 
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
    )

    # Start training
    print("start training ...")
    trainer.fit(model=task, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    
    # Save the trained model
    torch.save(task.state_dict(), os.path.join(out_folder, "trained_model.pth"))
    print(f"Model saved to {os.path.join(out_folder, 'trained_model.pth')}")
    return(num_bands, num_classes)

# ------------------------- Segmentation -------------------------

def prediction(image_path, model_path, output_path, num_bands, num_classes, patch_size, stride=32):
    """Apply the trained model on new data."""
    # Load model
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        in_channels=num_bands,
        num_classes=num_classes,
    )

    if torch.backends.mps.is_available():
        task.to('mps')
    task.load_state_dict(torch.load(model_path))
    task.eval()

    # Load image
    with rasterio.open(image_path) as src:
        image = src.read(out_dtype="float32")
        transform = src.transform
        width, height = src.width, src.height
        crs = src.crs

    # Transform into Tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()

    # Process patch-wise
    prediction_map = np.zeros((height, width), dtype=np.uint8)
    
    for row in tqdm(range(0, height, stride)):
        for col in range(0, width, stride):
            row_end = min(row + patch_size, height)
            col_end = min(col + patch_size, width)
            
            patch = image_tensor[:, :, row:row_end, col:col_end]

            pad_h = patch_size - patch.shape[2]
            pad_w = patch_size - patch.shape[3]
            # Skip empty patches
            if patch.shape[2] == 0 or patch.shape[3] == 0 or pad_h > patch_size/2 or pad_w > patch_size/2:
                continue
            # Process incomplete patches
            elif pad_h > 0 or pad_w > 0:
                padding = (0, max(0, pad_w), 0, max(0, pad_h))
                patch = F.pad(patch, padding, mode='reflect')
            
            pred = task(patch).argmax(dim=1).squeeze().byte().numpy()
            prediction_map[row:row_end, col:col_end] = pred[:row_end - row, :col_end - col]

    # Save result
    with rasterio.open(output_path, "w", driver="GTiff", height=height, width=width, count=1, dtype="uint8",
                       crs=crs, transform=transform) as dst:
        dst.write(prediction_map, 1)

    print(f"Saved prediction: {output_path}")


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
            displayName="Input Training Image Layer",
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
        param2.value = 64

        param3 = arcpy.Parameter(
            displayName="Number of Epochs",
            name="epochs",
            datatype="GPLong",
            parameterType="Required",
            direction="Input",
        )
        param3.value = 15

        param4 = arcpy.Parameter(
            displayName="Test Image Layer",
            name="test_image",
            datatype="GPRasterLayer",
            parameterType="Required",
            direction="Input",
            #multiValue=True,
        )

        return [param0, param01, param1, param2, param3, param4]

    def execute(self, parameters, messages):
        """Main execution logic of the tool."""        
        #image_layers = parameters[0].values  # List of input raster layers
        image_layers = parameters[0].value  # For only 1 possible input
        mask_layer = parameters[1].value
        out_folder = parameters[2].valueAsText
        batch_size = parameters[3].value
        epochs = parameters[4].value
        test_image = parameters[5].value
        patch_size = 64
        num_workers = 0

        # ------------------------- Run -------------------------
        in_image = arcpy.Describe(image_layers).catalogPath
        in_mask = arcpy.Describe(mask_layer).catalogPath
        os.makedirs(out_folder, exist_ok=True)
        processed_mask, index_to_class = preprocess_mask(in_mask, out_folder)
        num_bands, num_classes = train_model(in_image, processed_mask, out_folder, batch_size, epochs, num_workers, patch_size)

        test_image = arcpy.Describe(test_image).catalogPath
        trained_model = os.path.join(out_folder, "trained_model.pth")
        output_prediction = os.path.join(out_folder, "prediction_output_raw.TIF")
    
        prediction(test_image, trained_model, output_prediction, num_bands, num_classes, patch_size)

        postprocessed_output = os.path.join(out_folder, "prediction.TIF")
        postprocess_prediction(output_prediction, postprocessed_output, index_to_class)

        # Load segmentation in Map
        project = arcpy.mp.ArcGISProject("CURRENT")
        map_view = project.activeMap
        map_view.addDataFromPath(postprocessed_output)
        messages.addMessage("Prediction Layer has been added to the map")