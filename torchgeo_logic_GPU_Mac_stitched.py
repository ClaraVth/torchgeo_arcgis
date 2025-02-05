import torch
from torch import Tensor
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.trainers.segmentation import SemanticSegmentationTask
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, stack_samples, GeoDataset
from torchgeo.transforms import AugmentationSequential
from torchgeo.models import ResNet50_Weights
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop
from typing import Dict, Optional, Any, Iterable, Mapping
from multiprocessing import Lock
from tqdm import tqdm
import kornia
from kornia.geometry.transform import warp_affine
from tqdm import tqdm

import rasterio
import numpy as np
import os

# ------------------------- Helper Classes -------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

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

def preprocess_mask(mask_path):
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
    processed_mask_path = "processed_mask.tif"
    meta.update({"dtype": "uint8", "count": 1})
    with rasterio.open(processed_mask_path, "w", **meta) as dst:
        dst.write(indexed_mask.astype(np.uint8), 1)

    print(f"Processed mask saved to {processed_mask_path}")
    return processed_mask_path, class_to_index, index_to_class


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
def train_model(image_path, mask_path, out_folder, batch_size, epochs, num_workers, patch_size=64):
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
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=custom_stack_samples, pin_memory=True, persistent_workers=True)
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
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, collate_fn=custom_stack_samples, pin_memory=True, persistent_workers=True)


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
    return(num_bands, num_classes, os.path.join(out_folder, 'trained_model.pth'))

# ------------------------- Segmentation -------------------------

def prediction(image_path, model_path, output_path, num_bands, num_classes, patch_size=64, stride=32):
    """Funktion zur Anwendung des trainierten Modells auf neue Daten."""
    # Load model
    task = SemanticSegmentationTask(model="unet", backbone="resnet50", in_channels=num_bands, num_classes=num_classes)
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
            #print(pad_h, pad_w)
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


# ------------------------- Run -------------------------
"""
in_files = ["data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B1.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B2.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B3.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B4.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B5.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B6.TIF", "data/LC08_L2SP_023032_20230831_20230911_02_T1_SR_B7.TIF"]
in_image = combine_bands(in_files)
"""
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Important for MacOS

    training_image = "data/medium_size/training_image.tif"
    in_mask = "data/medium_size/mask.tif"
    out_folder = "."
    batch_size = 64
    epochs = 30
    num_workers = 8

    # Preprocess the mask
    processed_mask, class_to_index, index_to_class = preprocess_mask(in_mask)

    num_bands, num_classes, trained_model_path = train_model(training_image, processed_mask, out_folder, batch_size, epochs, num_workers)

    test_image = "data/medium_size/test_image.tif"
    trained_model = "./trained_model.pth"
    output_prediction = "output/prediction_output_raw.TIF"
    
    prediction(test_image, trained_model, output_prediction, num_bands, num_classes, num_workers)
    #prediction(test_image, trained_model, output_prediction, 7, 7)

    postprocessed_output = "output/30_final_prediction_patch64_length10000_1per_pretrained.TIF"
    #postprocessed_output = "output/test.TIF"
    postprocess_prediction(output_prediction, postprocessed_output, index_to_class)
