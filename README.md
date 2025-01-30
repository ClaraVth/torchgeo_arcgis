# torchgeo_arcgis
Implementation of TorchGeo into ArcGIS Pro

## Set up the environment
ArcGIS Pro needs to be installed. It does not work on macOS! I use version 3.4

Create a virtual environment with Python 3.11 and install the following packages:
```sh
conda create -n NAME python=3.11
```
```sh
conda activate NAME
```
```sh
conda install arcpy=3.4 -c esri
conda install pytorch torchvision  -c pytorch
pip install torchgeo
```

In ArcGIS Pro:
- Set the virtual environment: Go to Settings > Package Manager > Environment Manager > Add existing environment > Activate in list
- Add the .pyt script to the ArcGIS Pro Project Folder
- Add Toolbox: Content Pane > Right-click on Toolboxes > Add Toolbox (Navigate to location on PC)

## Current status
The .pyt file has been created and works in the environment of ArcGIS Pro. AAfter the first run, it seems to retain some memory, which eventually causes the program to crash. Therefore, I have isolated the code for semantic segmentation into the torchgeo_logic.py file. So far, I can run the script successfully and obtain an output GeoTIFF file with the segmentation. Comparing these results with the ground truth of this area reveals that they are incorrect. It recognizes the dominant classes but cannot assign them to the correct areas.

This is not an issue with the ArcGIS implementation. Therefore, the error must be within torchgeo_logic.py.

## Example Usage
I downloaded the Landsat 7, Landsat 8, and CDL datasets for testing from https://huggingface.co/datasets/torchgeo/tutorials/tree/main.
