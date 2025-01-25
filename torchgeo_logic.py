import torch
from torch import Tensor
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, stack_samples, GeoDataset
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Iterable, Mapping

import rasterio
import numpy as np
import os

# ------------------------- Helper Classes -------------------------
class DropFrozenKeys:
    def __call__(self, sample):
        sample.pop('crs')
        bbox = sample['bounds']
        sample['bounds'] = torch.tensor([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
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

    print(f"Multispectral image: {out_file}")
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


# ------------------------- Tool Definitions -------------------------
# ------------------------- Training -------------------------
def train_model(image_path, mask_path, out_folder, batch_size, epochs, patch_size=64):
    """
    Args:
        image_path (str): Path to input image.
        mask_path (str): Path to the mask.
        out_folder (str): Output folder.
        batch_size (int): Batch size.
        epochs (int): Number of Training epochs.
        patch_size (int): Path size for the model.
    """
    os.makedirs(out_folder, exist_ok=True)

    transforms = DropFrozenKeys()

    # Initialize datasets
    image_dataset = RasterDataset(paths=image_path, transforms=transforms)
    mask_dataset = RasterDataset(paths=mask_path, transforms=transforms)
    mask_dataset.is_image = False

    # combine datasets
    dataset = image_dataset & mask_dataset

    

    # Configure Sampler and DataLoader
    sampler = RandomGeoSampler(dataset, size=patch_size)    #, stride=patch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=custom_stack_samples)
    for batch in dataloader:
        x, y = batch["image"], batch["mask"]
        print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        break


    # Extract number of classes from mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        num_classes = len(np.unique(mask_data))
        #print(f"Number of classes: {num_classes}")
    
    with rasterio.open(image_path) as src:
        num_bands = src.count
        #print(f"Number of bands: {num_bands}")

    # Configure the model
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        in_channels=num_bands,
        num_classes=255, #num_classes,
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

    val_sampler = RandomGeoSampler(dataset, size=patch_size)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0, collate_fn=custom_stack_samples)


    # Configure Trainer 
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    print("start training ...")
    trainer.fit(model=task, train_dataloaders=dataloader, val_dataloaders=val_dataloader) #, datamodule=data_module
    
    # Save the trained model
    torch.save(task.state_dict(), os.path.join(out_folder, "trained_model.pth"))
    print(f"Model saved to {os.path.join(out_folder, 'trained_model.pth')}")
    return(num_bands, num_classes, os.path.join(out_folder, 'trained_model.pth'))

# ------------------------- Segmentation -------------------------

def prediction(image_path, model_path, output_path, num_bands, patch_size=64):
    """
    Applies a trained model to new data for prediction using Trainer.predict.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model (.pth file).
        output_path (str): Path to save the predicted output.
        patch_size (int): Size of the patches for inference.

    Returns:
        None
    """
    # Load the trained model
    task = SemanticSegmentationTask(
        model="unet",
        backbone="resnet50",
        in_channels=num_bands,  # Adjust to match input channels
        num_classes=255,  # Adjust to match the number of classes
    )
    task.load_state_dict(torch.load(model_path))
    task.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    # Extract metadata from the input image
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "uint8"})  # Update for single-band output

    # Prepare the input image
    transforms = DropFrozenKeys()
    image_dataset = RasterDataset(paths=image_path, transforms=transforms)

    sampler = GridGeoSampler(image_dataset, size=patch_size, stride=patch_size)
    dataloader = DataLoader(image_dataset, batch_size=1, sampler=sampler, num_workers=0)

    # Initialize the trainer
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Perform predictions
    print("Starting predictions...")
    predictions = trainer.predict(task, dataloaders=dataloader)

    # Save the predictions
    with rasterio.open(output_path, "w", **meta) as dst:
        print(f"Saving predictions to {output_path}...")
        for i, y_pred in enumerate(predictions):
            # Process the output
            y_pred = torch.argmax(y_pred, dim=1).squeeze(0).cpu().numpy()
            # Write each patch to the output file
            dst.write(y_pred.astype(np.uint8), 1)
        print(f"Saved prediction to {output_path}.")

# ------------------------- Application -------------------------

#in_files = ["data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B2.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B3.TIF"]
#in_image = combine_bands(in_files)
in_image = "data/training_image_cropped.tif"
in_mask = "data/2023_30m_cdls.tif"
out_folder = "."
batch_size = 8
epochs = 3

#num_bands, num_classes, trained_model_path = train_model(in_image, in_mask, out_folder, batch_size, epochs)

test_image = "data/test_image_cropped.tif"
trained_model = "./trained_model.pth"
output_prediction = "output/prediction_output.TIF"
prediction(test_image, trained_model, output_prediction, 3)